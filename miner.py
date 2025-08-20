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
        self.subtensor = bt.subtensor(network=NETWORK)
        
        # Get hotkey for encryption and URL
        self.hotkey = self.wallet.hotkey.ss58_address
        logger.info(f"Initialized miner with hotkey: {self.hotkey}")
        
        # Initialize timelock
        self.tlock = Timelock(DRAND_PUBLIC_KEY)
        
        # Validate hotkey is registered
        self._validate_registration()
    
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
        
        embeddings = []
        for asset in ASSETS:
            dim = ASSET_EMBEDDING_DIMS[asset]
            
            # TODO: Replace this with your actual model/prediction logic
            # This is a placeholder that generates random embeddings
            embedding = np.random.uniform(-1, 1, size=dim).tolist()
            
            # Validate embedding values are in [-1, 1] range
            embedding = [max(-1.0, min(1.0, val)) for val in embedding]
            
            embeddings.append(embedding)
            logger.debug(f"Generated {dim}-dim embedding for {asset}")
        
        logger.info(f"Generated embeddings for {len(ASSETS)} assets")
        return embeddings
    
    async def copy_other_miner_data(self, netuid: int, mg: bt.metagraph = None) -> List[List[float]]:
        """
        Copy other miner data from the public URL.
        """
        logger.info("Copying other miner data...")
        if mg is None:
            mg = bt.metagraph(netuid=netuid, network=NETWORK, sync=True)
        
        commits = self.subtensor.get_all_commitments(netuid)
        uid2hot = dict(zip(mg.uids.tolist(), mg.hotkeys))

        hotkey = uid2hot.get(netuid)

        object_url = commits.get(hotkey) if hotkey else None
        if not object_url:
            return

        try:
            parsed_url = urlparse(object_url)
            path = parsed_url.path

            if path.endswith('/'):
                logger.warning(f"UID {netuid} commit URL must not be a directory: {object_url}")
                return
            
            path_parts = path.lstrip('/').split('/')
            if len(path_parts) != 1:
                logger.warning(f"UID {netuid} commit URL must only contain the hotkey as the path: {object_url}")
                return

            object_name = path_parts[0]
            if object_name.lower() != (hotkey or "").lower():
                logger.warning(
                    f"UID {netuid} commit URL filename '{object_name}' does not match hotkey"
                )
                return
                
        except Exception as e:
            logger.warning(f"UID {netuid} commit URL validation failed for {object_url}: {e}")
            return

        payload_raw = await comms.download(object_url, max_size_bytes=MAX_PAYLOAD_BYTES)
        return self.decrypt_payload(payload_raw)
    
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
            future_time = time.time() + 30
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
        try:
            url = f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/rounds/{round_num}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"✅ Signature fetched for round {round_num}")
            return bytes.fromhex(data["signature"])
        except Exception as e:
            logger.error(f"Failed to fetch signature for round {round_num}: {e}")
            raise
    
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
            pt_bytes = self.tlock.tld(bytes.fromhex(ciphertext_hex), signature)
            
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
            expected_hotkey_norm = (self.hotkey or "").strip()
            if payload_hotkey_norm.lower() != expected_hotkey_norm.lower():
                raise ValueError(
                    f"Hotkey mismatch. Expected {expected_hotkey_norm[:8]}, got {payload_hotkey_norm[:8]}"
                )
            
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
        Upload payload file to public URL.
        
        Args:
            filepath: Path to local payload file
            
        Returns:
            Public URL where file is accessible
        """
        # This is a placeholder - implement based on your hosting solution
        # Examples: R2 bucket, personal server, GitHub gist, etc.
        
        public_url = f"{self.public_url_base}/{self.hotkey}"
        
        # TODO: Implement actual upload logic here
        # For R2 bucket example:
        # import boto3
        # s3 = boto3.client('s3', ...)
        # s3.upload_file(filepath, 'bucket-name', self.hotkey)
        
        # For now, just log the expected URL
        logger.info(f"Payload should be uploaded to: {public_url}")
        logger.warning("Upload functionality not implemented - please implement based on your hosting solution")
        
        return public_url
    
    def commit_url_to_subnet(self, public_url: str):
        """
        Commit the public URL to the Bittensor subnet.
        
        Args:
            public_url: Public URL where payload is accessible
        """
        try:
            logger.info(f"Committing URL to subnet: {public_url}")
            
            # Commit the URL on-chain
            self.subtensor.commit(
                wallet=self.wallet,
                netuid=NETUID,
                data=public_url
            )
            
            logger.info("Successfully committed URL to subnet")
            
        except Exception as e:
            logger.error(f"Failed to commit URL to subnet: {e}")
            raise
    
    def run_mining_cycle(self):
        """Run one complete mining cycle."""
        try:
            # Step 1: Generate embeddings
            embeddings = self.generate_embeddings()
            
            # Step 2: Encrypt payload
            payload = self.encrypt_payload(embeddings)
            
            # Step 3: Save payload file
            filename = self.hotkey
            filepath = self.save_payload(payload, filename)
            
            # Step 4: Upload to public URL
            public_url = self.upload_to_public_url(filepath)
            
            # Step 5: Commit URL to subnet (only if not commit_only mode)
            if not self.commit_only:
                self.commit_url_to_subnet(public_url)
            
            logger.info("Mining cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Mining cycle failed: {e}")
            raise
    
    def run_continuous_mining(self, interval_seconds: int = 60):
        """
        Run continuous mining with specified interval.
        
        Args:
            interval_seconds: Time between mining cycles
        """
        logger.info(f"Starting continuous mining with {interval_seconds}s intervals")
        
        try:
            while True:
                start_time = time.time()
                
                # Run mining cycle
                self.run_mining_cycle()
                
                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(0, interval_seconds - elapsed)
                
                logger.info(f"Cycle completed in {elapsed:.1f}s, sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Mining stopped by user")
        except Exception as e:
            logger.error(f"Continuous mining failed: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MANTIS Miner")
    parser.add_argument("--wallet.name", required=True, help="Bittensor wallet name")
    parser.add_argument("--wallet.hotkey", required=True, help="Bittensor hotkey name")
    parser.add_argument("--public-url", required=True, help="Base URL for hosting payload files")
    parser.add_argument("--commit-only", action="store_true", help="Only commit URL and exit")
    parser.add_argument("--continuous", action="store_true", help="Run continuous mining")
    parser.add_argument("--interval", type=int, default=60, help="Interval between cycles in seconds")
    
    args = parser.parse_args()
    
    try:
        # Initialize miner
        miner = MantisMiner(
            wallet_name=getattr(args, "wallet.name"),
            hotkey_name=getattr(args, "wallet.hotkey"),
            public_url_base=args.public_url,
            commit_only=args.commit_only
        )
        
        if args.commit_only:
            # Only commit URL
            public_url = f"{args.public_url.rstrip('/')}/{miner.hotkey}"
            miner.commit_url_to_subnet(public_url)
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