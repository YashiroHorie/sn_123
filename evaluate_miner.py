
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
import numpy as np
from dotenv import load_dotenv

import config
from cycle import get_miner_payloads
from model import salience as sal_fn
from storage import DataLog

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "main.log"), mode="a"),
    ],
)

weights_logger = logging.getLogger("weights")
weights_logger.setLevel(logging.DEBUG)
weights_logger.addHandler(
    logging.FileHandler(os.path.join(LOG_DIR, "weights.log"), mode="a")
)

for noisy in ("websockets", "aiohttp"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

load_dotenv()

os.makedirs(config.STORAGE_DIR, exist_ok=True)
DATALOG_PATH = os.path.join(config.STORAGE_DIR, "mantis_datalog.pkl.gz")
SAVE_INTERVAL = 120

async def get_asset_prices(session: aiohttp.ClientSession) -> dict[str, float] | None:
    try:
        async with session.get(config.PRICE_DATA_URL) as resp:
            resp.raise_for_status()
            text = await resp.text()
            data = json.loads(text)
            prices = data.get("prices", {})
            return prices
    except Exception as e:
        logging.error(f"Failed to fetch prices from {config.PRICE_DATA_URL}: {e}")
        return {}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", default="finney")
    p.add_argument("--netuid", type=int, default=config.NETUID)
    p.add_argument(
        "--no_download_datalog", 
        action="store_true", 
        help="Start with a fresh datalog instead of downloading from the archive."
    )
    p.add_argument(
        "--do_save",
        action="store_true",
        default=False,
        help="Whether to save the datalog periodically."
    )
    p.add_argument(
        "--archive",
        default=None,
        help="Path to local datalog .pkl.gz (defaults to STORAGE_DIR/mantis_datalog.pkl.gz)"
    )
    p.add_argument(
        "--prefer_local",
        action="store_true",
        help="If set, load existing local archive if it exists; otherwise download"
    )
    args = p.parse_args()

    while True:
        try:
            sub = bt.subtensor(network=args.network)
            mg = bt.metagraph(netuid=args.netuid, network=args.network, sync=True)
            break
        except Exception as e:
            logging.exception("Subtensor connect failed")
            time.sleep(30)
            continue

    if args.no_download_datalog:
        logging.info("`--no_download_datalog` flag set. Starting with a new, empty DataLog.")
        datalog = DataLog()
    else:
        archive_path = args.archive or DATALOG_PATH
        if args.prefer_local and os.path.exists(archive_path):
            logging.info("Loading local datalog from %s", archive_path)
            try:
                with gzip.open(archive_path, "rb") as f:
                    datalog = pickle.load(f)
                try:
                    datalog.raw_payloads = {}
                    logging.info("Pruned raw_payloads after local load (prefer_local).")
                except Exception:
                    pass
            except Exception as e:
                logging.warning("Local load failed (%s). Falling back to download.", e)
                datalog = DataLog.load(archive_path)
        else:
            datalog = DataLog.load(archive_path)
        
    stop_event = asyncio.Event()

    try:
        asyncio.run(run_main_loop(args, sub, mg, datalog, stop_event))
    except KeyboardInterrupt:
        logging.info("Exit signal received. Shutting down.")
    finally:
        stop_event.set()
        logging.info("Shutdown complete.")


async def decrypt_loop(datalog: DataLog, mg: bt.metagraph, stop_event: asyncio.Event):
    logging.info("Decryption loop started.")
    while not stop_event.is_set():
        try:
            uid_to_hotkey = dict(zip(mg.uids.tolist(), mg.hotkeys))
            await datalog.process_pending_payloads(uid_to_hotkey=uid_to_hotkey)
        except asyncio.CancelledError:
            break
        except Exception:
            logging.exception("An error occurred in the decryption loop.")
        await asyncio.sleep(5)
    logging.info("Decryption loop stopped.")


async def save_loop(datalog: DataLog, do_save: bool, stop_event: asyncio.Event):
    if not do_save:
        logging.info("DO_SAVE is False, skipping periodic saves.")
        return
    logging.info("Save loop started.")
    save_interval_seconds = SAVE_INTERVAL * 12
    while not stop_event.is_set():
        try:
            await asyncio.sleep(save_interval_seconds)
            logging.info("Initiating periodic datalog save...")
            await datalog.save(DATALOG_PATH)
        except asyncio.CancelledError:
            break
        except Exception:
            logging.exception("An error occurred in the save loop.")
    logging.info("⏹️ Save loop stopped.")


subtensor_lock = threading.Lock()

async def run_main_loop(
    args: argparse.Namespace,
    sub: bt.subtensor,
    mg: bt.metagraph,
    datalog: DataLog,
    stop_event: asyncio.Event,
    evaluate: bool = True
):
    last_block = sub.get_current_block()
    weight_thread: threading.Thread | None = None

    tasks = [
        asyncio.create_task(decrypt_loop(datalog, mg, stop_event)),
        asyncio.create_task(save_loop(datalog, args.do_save, stop_event)),
    ]

    async with aiohttp.ClientSession() as session:
        while not stop_event.is_set():
            try:
                with subtensor_lock:
                    current_block = sub.get_current_block()
                
                if current_block == last_block:
                    await asyncio.sleep(1)
                    continue
                last_block = current_block

                if current_block % config.SAMPLE_STEP != 0:
                    continue
                logging.info(f"Sampled block {current_block}")

                if current_block % 100 == 0:
                    with subtensor_lock:
                        mg.sync(subtensor=sub)
                    logging.info("Metagraph synced.")
                    if (evaluate):
                        uids_list = mg.uids.tolist() if isinstance(mg.uids, np.ndarray) else list(mg.uids)
                        hotkeys_list = mg.hotkeys.tolist() if isinstance(mg.hotkeys, np.ndarray) else list(mg.hotkeys)
                        uids_list.append(256)
                        hotkeys_list.append("5CXSWwg7jb19KuKJ9nNxJDwuZDe1bDADhaRYEJbSAtxeTbkG")
                        await datalog.sync_miners(dict(zip(uids_list, hotkeys_list)))
                    else:
                        await datalog.sync_miners(dict(zip(mg.uids.tolist(), mg.hotkeys)))
                    datalog.compute_and_display_uid_ages()

                asset_prices = await get_asset_prices(session)
                if not asset_prices:
                    logging.error("Failed to fetch prices for required assets.")
                    continue

                payloads = await get_miner_payloads(netuid=args.netuid, mg=mg, evaluate=evaluate)
                await datalog.append_step(current_block, asset_prices, payloads)

                if (
                    current_block % config.TASK_INTERVAL == 0
                    and (weight_thread is None or not weight_thread.is_alive())
                    and len(datalog.blocks) >= config.LAG * 2 + 1
                ):
                    def worker(training_data, uid_ages, block_snapshot, metagraph, cli_args):
                        if not training_data:
                            weights_logger.warning("Not enough data for salience, but checking for young UIDs.")
                        
                        weights_logger.info(f"=== Starting weight calculation | block {block_snapshot} ===")
                        
                        weights_logger.info("Calculating general salience...")
                        general_sal = sal_fn(training_data) if training_data else {}
                        if not general_sal:
                            weights_logger.info("General salience computation returned empty.")

                        weights_logger.info("Using general salience only (10-day skipped).")
                        sal = general_sal

                        if not sal:
                            weights_logger.info("Combined salience is empty. Assigning weights only to young UIDs if any.")
                            
                        uids = metagraph.uids.tolist()
                            
                       
                        young_uids = {uid for uid, age in uid_ages.items() if age < 32000}
                        weights_logger.info(f"Found {len(young_uids)} young UIDs (<32000 blocks).")
                        
                        fixed_weight_per_young_uid = 0.0001
                        active_young_uids = {uid for uid in young_uids if uid in uids}
                        
                       
                       
                        final_weights = {uid: float(sal.get(uid, 0.0)) for uid in uids}
                        
                       
                        for uid in active_young_uids:
                            final_weights[uid] = final_weights.get(uid, 0.0) + fixed_weight_per_young_uid
                            
                        if not final_weights:
                            weights_logger.warning("No weights to set after processing.")
                            return
                            
                        total_weight = sum(final_weights.values())
                        if total_weight <= 0:
                            weights_logger.warning("Total calculated weight is zero or negative, skipping set.")
                            return
                            
                        normalized_weights = {uid: w / total_weight for uid, w in final_weights.items()}
                            
                        w = torch.tensor([normalized_weights.get(uid, 0.0) for uid in uids], dtype=torch.float32)
                            
                        if w.sum() > 0:
                            final_w = w / w.sum()
                        else:
                            weights_logger.warning("Zero-sum weights tensor, skipping set.")
                            return
                        
                        weights_to_log = {uid: f"{weight:.8f}" for uid, weight in normalized_weights.items() if uid in uids and weight > 0}
                        weights_logger.info(f"Normalized weights for block {block_snapshot}: {json.dumps(weights_to_log)}")
                        weights_logger.info(f"Final tensor sum before setting weights: {final_w.sum().item()}")
                        
                        # try:
                        #     thread_sub = bt.subtensor(network=cli_args.network)
                        #     thread_wallet = bt.wallet(
                        #         name=getattr(cli_args, "wallet.name"), 
                        #         hotkey=getattr(cli_args, "wallet.hotkey")
                        #     )
                        #     thread_sub.set_weights(
                        #         netuid=cli_args.netuid, wallet=thread_wallet,
                        #         uids=metagraph.uids, weights=final_w,
                        #         wait_for_inclusion=False,
                        #     )
                        #     weights_logger.info(f"Weights set at block {block_snapshot} (max={final_w.max():.4f})")
                        # except Exception as e:
                        #     weights_logger.error(f"Failed to set weights: {e}", exc_info=True)

                    max_block_for_training = current_block - config.TASK_INTERVAL
                    async with datalog._lock:
                        training_data_copy = copy.deepcopy(datalog.get_training_data(max_block_number=max_block_for_training))
                        uid_ages_copy = copy.deepcopy(datalog.uid_age_in_blocks)

                    weight_thread = threading.Thread(
                        target=worker,
                        args=(training_data_copy, uid_ages_copy, current_block, copy.deepcopy(mg), copy.deepcopy(args)),
                        daemon=True,
                    )
                    weight_thread.start()

            except KeyboardInterrupt:
                stop_event.set()
            except Exception:
                logging.error("Error in main loop", exc_info=True)
                await asyncio.sleep(10)
    
    logging.info("Main loop finished. Cleaning up background tasks.")
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    main()
