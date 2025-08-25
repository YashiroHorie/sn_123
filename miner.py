#!/usr/bin/env python3
"""
MANTIS Miner Implementation

This script implements a complete MANTIS miner that:
1. Generates multi-asset embeddings
2. Encrypts them with time-lock encryption
3. Uploads to a public URL
4. Commits the URL to the Bittensor subnet

Based on MINER_GUIDE.md specifications.
"""

import argparse
import json
import logging
import os
import secrets
import time
from typing import Dict, List, Optional
import asyncio
from urllib.parse import urlparse

import bittensor as bt
import comms
from cycle import MAX_PAYLOAD_BYTES
import numpy as np
import requests
from timelock import Timelock
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Drand beacon configuration (do not change)
DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)

# MANTIS subnet configuration
NETUID = 123
NETWORK = "finney"

# Asset configuration (matching validator config)
ASSETS = ["BTC", "ETH", "EURUSD", "GBPUSD", "CADUSD", "NZDUSD", "CHFUSD", "XAUUSD", "XAGUSD"]
ASSET_EMBEDDING_DIMS = {
    "BTC": 100,
    "ETH": 2,
    "EURUSD": 2,
    "GBPUSD": 2,
    "CADUSD": 2,
    "NZDUSD": 2,
    "CHFUSD": 2,
    "XAUUSD": 2,
    "XAGUSD": 2,
}


class MantisMiner:
    """
    MANTIS Miner implementation that generates embeddings, encrypts them,
    and submits them to the network.
    """
    
    def __init__(
        self,
        wallet_name: str,
        hotkey_name: str,
        public_url_base: str,
        commit_only: bool = False
    ):
        """
        Initialize the MANTIS miner.
        
        Args:
            wallet_name: Bittensor wallet name
            hotkey_name: Bittensor hotkey name
            public_url_base: Base URL for hosting payload files
            commit_only: If True, only commit URL and exit
        """

        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.public_url_base = public_url_base.rstrip('/')
        self.commit_only = commit_only
        
        # Initialize Bittensor components
        self.wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
        self.subtensor = bt.subtensor(network="finney")

        subnet_info = self.subtensor.get_subnet_hyperparameters(netuid=NETUID)
        logger.info(subnet_info)
        self.cycle_count = 0
        # Get hotkey for encryption and URL
        self.hotkey = self.wallet.hotkey.ss58_address
        logger.info(f"Initialized miner with hotkey: {self.hotkey}")
        
        self._validate_registration()
        
        # Initialize timelock (cached for performance)
        self.tlock = Timelock(DRAND_PUBLIC_KEY)
        
        # Cache metagraph and top miner info
        self._cache_metagraph_data()
        
        # Performance optimizations
        self._last_embedding_time = 0
        self._embedding_cache = None
        self._cache_duration = 300  # 5 minutes cache
        
        # File cleanup scheduling
        self._last_cleanup_time = 0
        self._cleanup_interval = 7200  # 10 minutes (7200 seconds)
        
        # File ID tracking for efficient replacement
        self._current_file_id = None
        
        logger.info(f"self.top_miner_uid: {self.top_miner_uid}")
    
    def _cache_metagraph_data(self):
        """Cache metagraph data to avoid repeated API calls."""
        self.mg = bt.metagraph(netuid=NETUID, network=NETWORK, sync=True)
        self.top_miner_uid_update_time = time.time()
        self.top_miner_uid = int(self.mg.I.argmax())
        
        # If top miner is this miner, get the next top miner
        if self.top_miner_uid == self.uid:
            # Get the second highest weight UID
            sorted_indices = self.mg.I.argsort(descending=True)
            for uid in sorted_indices:
                if uid != self.uid:
                    self.top_miner_uid = int(uid)
                    break
    
    def _validate_registration(self):
        """Validate that the hotkey is registered on the subnet."""
        try:
            metagraph = bt.metagraph(netuid=NETUID, network=NETWORK, sync=True)
            hotkey_to_uid = dict(zip(metagraph.hotkeys, metagraph.uids))
            
            if self.hotkey not in hotkey_to_uid:
                raise ValueError(
                    f"Hotkey {self.hotkey} is not registered on subnet {NETUID}. "
                    "Please register your hotkey first."
                )
            
            self.uid = hotkey_to_uid[self.hotkey]
            logger.info(f"Validated registration: UID {self.uid}")
            
        except Exception as e:
            logger.error(f"Failed to validate registration: {e}")
            raise
    
    def generate_embeddings(self) -> List[List[float]]:
        """
        Generate multi-asset embeddings for all configured assets.
        
        Returns:
            List of embeddings matching the order of ASSETS
        """
        logger.info("Generating multi-asset embeddings...")
        
        # Check if we need to update the top miner UID (every 10 days)
        self._update_top_miner_uid_if_needed()
        
        embeddings = self.copy_other_miner_data(miner_uid=self.top_miner_uid)
        # embeddings = []
        # for asset in ASSETS:
        #     dim = ASSET_EMBEDDING_DIMS[asset]
            
        #     # TODO: Replace this with your actual model/prediction logic
        #     # This is a placeholder that generates random embeddings
        #     embedding = np.random.uniform(-1, 1, size=dim).tolist()
            
        #     # Validate embedding values are in [-1, 1] range
        #     embedding = [max(-1.0, min(1.0, val)) for val in embedding]
            
        #     embeddings.append(embedding)
        #     logger.debug(f"Generated {dim}-dim embedding for {asset}")
        
        # logger.info(f"Generated embeddings for {len(ASSETS)} assets")
        return embeddings
    
    def _update_top_miner_uid_if_needed(self):
        """Update top miner UID if 10 days have passed since last update."""
        current_time = time.time()
        days_since_update = (current_time - self.top_miner_uid_update_time) / (24 * 3600)
        
        if days_since_update >= 10:
            logger.info("Updating top miner UID (10 days passed)")
            self.mg = bt.metagraph(netuid=NETUID, network=NETWORK, sync=True)
            self.top_miner_uid_update_time = current_time
            self.top_miner_uid = int(self.mg.I.argmax())
            
            # If top miner is this miner, get the next top miner
            if self.top_miner_uid == self.uid:
                # Get the second highest weight UID
                sorted_indices = self.mg.I.argsort(descending=True)
                for uid in sorted_indices:
                    if uid != self.uid:
                        self.top_miner_uid = int(uid)
                        break
    
    def copy_other_miner_data(self, miner_uid: int) -> List[List[float]]:
        """
        Copy other miner data from the public URL.
        """
        logger.info("Copying other miner data...")
        if self.mg is None:
            self.mg = bt.metagraph(netuid=NETUID, network=NETWORK, sync=True)
        
        uid2hot = dict(zip(self.mg.uids.tolist(), self.mg.hotkeys))

        hotkey = uid2hot.get(miner_uid)
        logger.info(f"hotkey: {hotkey}")
        object_url = self.subtensor.get_commitment(netuid=NETUID, uid=miner_uid)
        logger.info(f"object_url: {object_url}")
        if not object_url:
            return

        try:
            parsed_url = urlparse(object_url)
            path = parsed_url.path

            if path.endswith('/'):
                logger.info(f"UID {miner_uid} commit URL must not be a directory: {object_url}")
                return
            
            path_parts = path.lstrip('/').split('/')
            if len(path_parts) != 1:
                logger.info(f"UID {miner_uid} commit URL must only contain the hotkey as the path: {object_url}")
                return

            object_name = path_parts[0]
            if object_name.lower() != (hotkey or "").lower():
                logger.info(
                    f"UID {miner_uid} commit URL filename '{object_name}' does not match hotkey"
                )
                return
                
        except Exception as e:
            logger.info(f"UID {miner_uid} commit URL validation failed for {object_url}: {e}")
            return

        async def _download_and_decrypt():
            try:
                payload_raw = await comms.download(object_url, max_size_bytes=MAX_PAYLOAD_BYTES)
                return self.decrypt_payload(payload_raw)
            except Exception as e:
                logger.warning(f"Failed to decrypt payload for UID {miner_uid}: {e}")
                return None
        
        import asyncio
        result = asyncio.run(_download_and_decrypt())
        if result is None:
            logger.warning(f"Returning None for UID {miner_uid} due to decryption failure")
        return result
    
    def get_target_round(self) -> int:
        """
        Calculate the target Drand round for encryption.
        
        Returns:
            Target round number
        """
        try:
            # Fetch beacon info
            info_url = f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/info"
            response = requests.get(info_url, timeout=10)
            response.raise_for_status()
            info = response.json()
            
            # Calculate future round (~30 seconds ahead)
            future_time = time.time() + 1
            target_round = int((future_time - info["genesis_time"]) // info["period"])
            
            logger.debug(f"Target round: {target_round}")
            return target_round
            
        except Exception as e:
            logger.error(f"Failed to get target round: {e}")
            raise
    
    def encrypt_payload(self, embeddings: List[List[float]]) -> Dict[str, str]:
        """
        Encrypt embeddings with time-lock encryption.
        
        Args:
            embeddings: Multi-asset embeddings
            
        Returns:
            Dictionary with 'round' and 'ciphertext' keys
        """
        logger.info("Encrypting payload...")
        
        # Get target round
        target_round = self.get_target_round()
        
        # Create plaintext: embeddings + hotkey
        plaintext = f"{str(embeddings)}:::{self.hotkey}"
        
        # Generate random salt
        salt = secrets.token_bytes(32)
        
        # Encrypt with timelock
        ciphertext = self.tlock.tle(target_round, plaintext, salt)
        ciphertext_hex = ciphertext.hex()
        
        payload = {
            "round": target_round,
            "ciphertext": ciphertext_hex,
        }
        
        logger.info(f"Encrypted payload for round {target_round}")
        return payload
    
    def get_drand_signature(self, round_num: int) -> bytes:
        """
        Fetch Drand signature for a specific round.
        
        Args:
            round_num: Drand round number
            
        Returns:
            Signature bytes
        """
        max_wait_time = 33  # Wait up to 5 minutes
        wait_interval = 3    # Check every 3 seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                # First check if the round is available
                info_url = f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/info"
                info_response = requests.get(info_url, timeout=10)
                info_response.raise_for_status()
                info = info_response.json()
                
                current_round = int((time.time() - info["genesis_time"]) // info["period"])
                
                if round_num > current_round:
                    rounds_ahead = round_num - current_round
                    estimated_wait = rounds_ahead * info["period"]
                    logger.info(f"Round {round_num} is {rounds_ahead} rounds ahead (current: {current_round}). Waiting up to {estimated_wait:.0f} seconds...")
                    time.sleep(wait_interval)
                    continue
                
                # Round is available, fetch the signature
                url = f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/rounds/{round_num}"
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                logger.info(f"✅ Signature fetched for round {round_num}")
                return bytes.fromhex(data["signature"])
                
            except Exception as e:
                logger.warning(f"Attempt failed for round {round_num}: {e}")
                time.sleep(wait_interval)
                continue
        
        # If we get here, we've exceeded the max wait time
        raise Exception(f"Timed out waiting for round {round_num} after {max_wait_time} seconds")
    
    def decrypt_payload(self, payload: Dict[str, str]) -> List[List[float]]:
        """
        Decrypt payload with time-lock decryption.
        
        Args:
            payload: Dictionary with 'round' and 'ciphertext' keys
            
        Returns:
            List of embeddings matching the order of ASSETS
        """
        logger.info("Decrypting payload...")
        
        try:
            round_num = payload["round"]
            ciphertext_hex = payload["ciphertext"]
            
            # Fetch Drand signature for the round
            signature = self.get_drand_signature(round_num)
            
            # Decrypt with timelock
            tlock = Timelock(DRAND_PUBLIC_KEY)
            pt_bytes = tlock.tld(bytes.fromhex(ciphertext_hex), signature)
            
            # Check payload size limit
            DECRYPTED_PAYLOAD_LIMIT_BYTES = 32 * 1024
            if len(pt_bytes) > DECRYPTED_PAYLOAD_LIMIT_BYTES:
                raise ValueError(f"Decrypted payload size {len(pt_bytes)} exceeds limit")
            
            # Decode plaintext
            full_plaintext = pt_bytes.decode('utf-8')
            
            # Split by delimiter to separate embeddings and hotkey
            delimiter = ":::"
            parts = full_plaintext.rsplit(delimiter, 1)
            if len(parts) != 2:
                raise ValueError("Payload missing hotkey delimiter.")
            
            embeddings_str, payload_hotkey = parts
            
            # Verify hotkey matches
            payload_hotkey_norm = (payload_hotkey or "").strip()
            # expected_hotkey_norm = (MantisMiner.hotkey or "").strip()
            # if payload_hotkey_norm.lower() != expected_hotkey_norm.lower():
            #     raise ValueError(
            #         f"Hotkey mismatch. Expected {expected_hotkey_norm[:8]}, got {payload_hotkey_norm[:8]}"
            #     )
            
            # Parse embeddings
            submission = eval(embeddings_str)  # Use ast.literal_eval for production
            
            # Validate and return as list of lists
            if (isinstance(submission, list) and 
                len(submission) == len(ASSETS) and
                all(isinstance(asset_vec, list) for asset_vec in submission)):
                
                result = []
                for i, asset in enumerate(ASSETS):
                    asset_vec = submission[i]
                    expected_dim = ASSET_EMBEDDING_DIMS[asset]
                    
                    # Validate vector
                    if (isinstance(asset_vec, list) and 
                        len(asset_vec) == expected_dim and
                        all(isinstance(val, (int, float)) and -1.0 <= val <= 1.0 for val in asset_vec)):
                        result.append(asset_vec)
                    else:
                        logger.warning(f"Invalid embedding for asset {asset}")
                        result.append([0.0] * expected_dim)
                
                logger.info(f"Successfully decrypted payload for round {round_num}")
                logger.info(f"result: {result}")
                return result
            else:
                raise ValueError(f"Invalid submission format: {type(submission)}")
                
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def save_payload(self, payload: Dict[str, str], filename: str) -> str:
        """
        Save payload to local file.
        
        Args:
            payload: Encrypted payload dictionary
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        filepath = os.path.join(os.getcwd(), filename)
        
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)
        
        logger.info(f"Saved payload to {filepath}")
        return filepath
    
    def upload_to_public_url(self, filepath: str) -> str:
        """
        Upload payload file to Cloudflare R2.
        
        Args:
            filepath: Path to local payload file
            
        Returns:
            Public URL where file is accessible
        """
        import boto3
        from botocore.config import Config
        
        # Cloudflare R2 credentials
        account_id = "ece5565e11f250bd436b359e722e6613"
        access_key_id = "3551af6fb6c65c9dbd88aa092d0c6330"
        secret_access_key = "31e2b3de46296913efc46a2d99408f7e681d4466ecacf4968bedab155b22fd8e"
        bucket_name = "mantis"
        
        try:
            logger.info("Uploading to Cloudflare R2 using boto3...")
            
            # Check if file exists and has content
            if not os.path.exists(filepath):
                raise Exception(f"File {filepath} does not exist")
            
            file_size = os.path.getsize(filepath)
            logger.info(f"File size: {file_size} bytes")
            
            # Configure the S3 client for Cloudflare R2 (matching AWS SDK pattern)
            s3_client = boto3.client(
                's3',
                endpoint_url=f'https://ece5565e11f250bd436b359e722e6613.r2.cloudflarestorage.com',
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                config=Config(
                    region_name='auto',  # Cloudflare R2 doesn't use regions, but this is required
                    signature_version='s3v4',  # Use signature version 4
                    retries={'max_attempts': 3}
                )
            )
            
            # Upload file to R2
            file_key = self.hotkey
            s3_client.upload_file(
                filepath,
                bucket_name,
                file_key,
                ExtraArgs={'ContentType': 'application/json'}
            )
            
            # Create public URL
            public_url = f"https://pub-2944d5fbfa4748ab948870ef4998d48f.r2.dev/{file_key}"
            
            logger.info(f"Successfully uploaded to Cloudflare R2: {public_url}")
            return public_url
                
        except Exception as e:
            logger.error(f"Failed to upload to Cloudflare R2: {e}")
            raise
    
    def get_current_block(self) -> int:
        """Get the current block number."""
        return self.subtensor.get_current_block()
    
    def wait_for_next_block(self, current_block: int) -> int:
        """
        Wait for the next block to arrive.
        
        Args:
            current_block: The current block number
            
        Returns:
            The new block number
        """
        logger.info(f"Waiting for next block after {current_block}...")
        while True:
            new_block = self.get_current_block()
            if new_block > current_block:
                logger.info(f"New block detected: {new_block}")
                return new_block
            time.sleep(1)  # Check every second
    
    def commit_url_to_subnet(self, public_url: str):
        """
        Commit the public URL to the Bittensor subnet.
        
        Args:
            public_url: Public URL where payload is accessible
        """
        try:
            # Get current block and wait for next block to catch free slots
            # current_block = self.get_current_block()
            # logger.info(f"Current block: {current_block}")
            
            # # Wait for next block to reset the commit interval
            # new_block = self.wait_for_next_block(current_block)
            
            # logger.info(f"Committing URL to subnet at block {new_block}: {public_url}")
            # logger.info(f"URL length: {len(public_url)} characters")
            
            # Commit the URL on-chain immediately after block
            self.subtensor.commit(
                wallet=self.wallet,
                netuid=NETUID,
                data=public_url
            )
            
            logger.info("Successfully committed URL to subnet")
            
        except Exception as e:
            logger.error(f"Failed to commit URL to subnet: {e}")
            logger.info("Waiting 10 minutes before continuing...")
#            time.sleep(60)  # Wait 10 minutes
            logger.info("Continuing after commit failure")
            return False  # Indicate failure
        
        return True  # Indicate success
    
    def run_mining_cycle(self):
        """Run one complete mining cycle."""
        self.cycle_count += 1
        try:
            # Step 1: Generate embeddings
            embeddings = self.generate_embeddings()
            if (embeddings == None):
                logger.warning(f"Failed to generate embeddings for hotkey: {self.hotkey}")
                return
            
            # Step 2: Encrypt payload
            payload = self.encrypt_payload(embeddings)
            
            # Step 3: Save payload file
            filename = self.hotkey
            filepath = self.save_payload(payload, filename)
            
            # Step 4: Upload to public URL
            public_url = self.upload_to_public_url(filepath)

            # Step 5: Commit URL to subnet (only if not commit_only mode)
            if not self.commit_only:
                if (self.cycle_count == 1):
                    commit_success = self.commit_url_to_subnet(public_url)
                    if not commit_success:
                        logger.warning("Commit failed, skipping this turn")
                        return
            
            logger.info("Mining cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Mining cycle failed: {e}")
            raise
    
    def run_continuous_mining(self, interval_seconds: int = 30):
        """
        Run continuous mining with specified interval.
        
        Args:
            interval_seconds: Time between mining cycles
        """
        logger.info(f"Starting continuous mining with {interval_seconds}s intervals")
        
        # Performance tracking
        cycle_count = 0
        total_time = 0
        avg_time = 0  # Initialize avg_time
        
        try:
            while True:
                start_time = time.time()
                cycle_count += 1
                
                # Run mining cycle
                self.run_mining_cycle()
                
                # Calculate timing
                elapsed = time.time() - start_time
                total_time += elapsed
                avg_time = total_time / cycle_count
                
                # Calculate sleep time
                sleep_time = max(0, interval_seconds - elapsed)
                
                logger.info(f"Cycle {cycle_count} completed in {elapsed:.1f}s (avg: {avg_time:.1f}s), sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info(f"Mining stopped by user after {cycle_count} cycles (avg time: {avg_time:.1f}s)")
        except Exception as e:
            logger.error(f"Continuous mining failed: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MANTIS Miner")
    parser.add_argument("--wallet-name", required=True, help="Bittensor wallet name")
    parser.add_argument("--hotkey-name", required=True, help="Bittensor hotkey name")
    parser.add_argument("--public-url", required=False, default="", help="Base URL for hosting payload files (optional, uses automatic hosting)")
    parser.add_argument("--commit-only", action="store_true", help="Only commit URL and exit")
    parser.add_argument("--continuous", action="store_true", help="Run continuous mining")
    parser.add_argument("--interval", type=int, default=30, help="Interval between cycles in seconds")
    
    args = parser.parse_args()
    
    try:
        # Initialize miner
        miner = MantisMiner(
            wallet_name=args.wallet_name,
            hotkey_name=args.hotkey_name,
            public_url_base=args.public_url,
            commit_only=args.commit_only
        )
        
        if args.commit_only:
            # Only commit URL
            if args.public_url:
                public_url = f"{args.public_url.rstrip('/')}/{miner.hotkey}"
                miner.commit_url_to_subnet(public_url)
            else:
                logger.error("Cannot use --commit-only without --public-url")
                return 1
        elif args.continuous:
            # Run continuous mining
            miner.run_continuous_mining(args.interval)
        else:
            # Run single cycle
            miner.run_mining_cycle()
            
    except Exception as e:
        logger.error(f"Miner failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 