#!/usr/bin/env python3
"""
Miner Evaluator

This script evaluates a specific miner by:
1. Fetching payloads from a fixed URL
2. Processing them like a validator
3. Using simplified datalog.append_step(asset_prices, payload)
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
import asyncio
import copy
import json
import gzip
import pickle

import bittensor as bt
import torch
import aiohttp
from dotenv import load_dotenv

import config
from model import salience as sal_fn
from storage import DataLog
import comms

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "evaluate_miner.log"), mode="a"),
    ],
)

weights_logger = logging.getLogger("weights")
weights_logger.setLevel(logging.DEBUG)
weights_logger.addHandler(
    logging.FileHandler(os.path.join(LOG_DIR, "evaluate_weights.log"), mode="a")
)

for noisy in ("websockets", "aiohttp"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

load_dotenv()

os.makedirs(config.STORAGE_DIR, exist_ok=True)
DATALOG_PATH = os.path.join(config.STORAGE_DIR, "evaluate_datalog.pkl.gz")
SAVE_INTERVAL = 120

async def get_asset_prices(session: aiohttp.ClientSession) -> dict[str, float] | None:
    """Fetch asset prices from the configured URL."""
    try:
        async with session.get(config.PRICE_DATA_URL) as resp:
            resp.raise_for_status()
            text = await resp.text()
            data = json.loads(text)
            prices = data.get("prices", {})
            logging.info(f"Fetched prices for {len(prices)} assets: {prices}")
            return prices
    except Exception as e:
        logging.error(f"Failed to fetch prices from {config.PRICE_DATA_URL}: {e}")
        return {}

async def get_fixed_payloads(fixed_url: str) -> dict[int, dict]:
    """
    Fetch payloads from a fixed URL instead of from all miners.
    
    Args:
        fixed_url: The URL to fetch payloads from
        
    Returns:
        Dictionary with a single entry for the fixed miner
    """
    payloads = {}
    
    try:
        # Download payload from fixed URL
        payload_raw = await comms.download(fixed_url, max_size_bytes=25 * 1024 * 1024)
        if payload_raw:
            # Use a dummy UID (999) for the fixed miner
            payloads[999] = payload_raw
            logging.info(f"Successfully downloaded payload from {fixed_url}")
        else:
            logging.warning(f"No payload data received from {fixed_url}")
    except Exception as e:
        logging.error(f"Failed to download payload from {fixed_url}: {e}")
    
    return payloads

async def decrypt_loop(datalog: DataLog, stop_event: asyncio.Event):
    """Background loop for decrypting payloads."""
    logging.info("🔓 Starting decrypt loop.")
    while not stop_event.is_set():
        try:
            await datalog.decrypt_step()
            await asyncio.sleep(1)
        except Exception as e:
            logging.error(f"Error in decrypt loop: {e}")
            await asyncio.sleep(5)
    logging.info("⏹️ Decrypt loop stopped.")

async def save_loop(datalog: DataLog, do_save: bool, stop_event: asyncio.Event):
    """Background loop for saving datalog."""
    logging.info("💾 Starting save loop.")
    while not stop_event.is_set():
        try:
            if do_save:
                await datalog.save(DATALOG_PATH)
            await asyncio.sleep(SAVE_INTERVAL)
        except Exception as e:
            logging.error(f"Error in save loop: {e}")
            await asyncio.sleep(10)
    logging.info("⏹️ Save loop stopped.")

async def main():
    p = argparse.ArgumentParser(description="Evaluate a specific miner using fixed URL")
    p.add_argument("--fixed-url", required=True, help="Fixed URL to fetch payloads from")
    p.add_argument("--network", default="finney")
    p.add_argument("--netuid", type=int, default=config.NETUID)
    args = p.parse_args()

    while True:
        try:
            sub = bt.subtensor(network=args.network)
            mg = bt.metagraph(netuid=args.netuid, network=args.network, sync=True)
            break
        except Exception as e:
            logging.error(f"Failed to initialize Bittensor components: {e}")
            time.sleep(5)

    # Initialize datalog
    datalog = DataLog()
    
    if args.archive is None:
        args.archive = DATALOG_PATH
    
    if not args.no_download_datalog:
        try:
            if args.prefer_local and os.path.exists(args.archive):
                logging.info(f"Loading local datalog from {args.archive}")
                datalog.load(args.archive)
            else:
                logging.info("Downloading datalog from archive...")
                datalog.download_and_load(config.DATALOG_ARCHIVE_URL, args.archive)
        except Exception as e:
            logging.warning(f"Failed to load datalog: {e}")
            logging.info("Starting with fresh datalog.")

    # Initialize metagraph data
    await datalog.sync_miners(dict(zip(mg.uids.tolist(), mg.hotkeys)))
    datalog.compute_and_display_uid_ages()

    # Main evaluation loop
    stop_event = asyncio.Event()
    
    tasks = [
        asyncio.create_task(decrypt_loop(datalog, stop_event)),
        asyncio.create_task(save_loop(datalog, args.do_save, stop_event)),
    ]

    async with aiohttp.ClientSession() as session:
        while not stop_event.is_set():
            try:
                current_block = sub.get_current_block()
                
                if current_block % config.SAMPLE_STEP != 0:
                    await asyncio.sleep(1)
                    continue
                    
                logging.info(f"Evaluating block {current_block}")

                # Fetch asset prices
                asset_prices = await get_asset_prices(session)
                if not asset_prices:
                    logging.error("Failed to fetch prices for required assets.")
                    await asyncio.sleep(5)
                    continue

                # Fetch payloads from fixed URL
                payloads = await get_fixed_payloads(args.fixed_url)
                
                # Use simplified append_step (without block number)
                await datalog.append_step(asset_prices, payloads)

                # Calculate weights periodically
                if (
                    current_block % config.TASK_INTERVAL == 0
                    and len(datalog.blocks) >= config.LAG * 2 + 1
                ):
                    logging.info("Calculating weights for evaluation...")
                    
                    # Get training data
                    max_block_for_training = current_block - config.TASK_INTERVAL
                    async with datalog._lock:
                        training_data = datalog.get_training_data(max_block_number=max_block_for_training)
                        uid_ages = datalog.uid_age_in_blocks

                    if training_data:
                        # Calculate salience
                        sal = sal_fn(training_data)
                        logging.info(f"Salience calculation completed. Top weights: {dict(list(sal.items())[:5])}")
                    else:
                        logging.warning("Not enough data for salience calculation.")

                await asyncio.sleep(1)

            except KeyboardInterrupt:
                stop_event.set()
            except Exception as e:
                logging.error(f"Error in evaluation loop: {e}", exc_info=True)
                await asyncio.sleep(10)
    
    logging.info("Evaluation loop finished. Cleaning up background tasks.")
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())
