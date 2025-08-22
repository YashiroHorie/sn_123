#!/usr/bin/env python3
"""
Miner Status Checker

This script checks the exact status of a specific miner (UID 237) on the MANTIS subnet
using the Bittensor SDK.
"""

import bittensor as bt
import json
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# MANTIS subnet configuration
NETUID = 123
NETWORK = "finney"
TARGET_HOTKEY = "5CXSWwg7jb19KuKJ9nNxJDwuZDe1bDADhaRYEJbSAtxeTbkG" # "5DXgzL7bdDtWaokxXC6b9NSk8DPD8LKWSr1rcH1mnAZDiBgV" 

def get_miner_status(hotkey: str = TARGET_HOTKEY) -> Dict[str, Any]:
    """
    Get comprehensive status information for a specific miner.
    
    Args:
        hotkey: The hotkey of the miner to check
        
    Returns:
        Dictionary containing miner status information
    """
    try:
        # Initialize Bittensor components
        subtensor = bt.subtensor(network=NETWORK)
        metagraph = bt.metagraph(netuid=NETUID, network=NETWORK, sync=True)
        
        # Check if hotkey exists in metagraph
        hotkeys_list = metagraph.hotkeys.tolist() if hasattr(metagraph.hotkeys, 'tolist') else list(metagraph.hotkeys)
        if hotkey not in hotkeys_list:
            return {
                "error": f"Hotkey {hotkey} not found in subnet {NETUID}",
                "registered_hotkeys": hotkeys_list
            }
        
        # Get hotkey index in metagraph
        hotkey_index = hotkeys_list.index(hotkey)
        
        # Get basic miner information
        uids_list = metagraph.uids.tolist() if hasattr(metagraph.uids, 'tolist') else list(metagraph.uids)
        stake_list = metagraph.S.tolist() if hasattr(metagraph.S, 'tolist') else list(metagraph.S)
        weight_list = metagraph.I.tolist() if hasattr(metagraph.I, 'tolist') else list(metagraph.I)
        
        uid = uids_list[hotkey_index]
        stake = stake_list[hotkey_index]
        weight = weight_list[hotkey_index]
        
        # Get current block
        current_block = subtensor.get_current_block()
        
        # Get commitment info
        commitments = subtensor.get_all_commitments(netuid=NETUID)
        commitment_url = commitments.get(hotkey)
        
        # Get subnet info
        subnet_info = subtensor.get_subnet_hyperparameters(netuid=NETUID)
        
        # Check if miner is active (has stake)
        is_active = stake > 0
        
        # Get rank information
        sorted_weights = sorted(weight_list, reverse=True)
        rank = sorted_weights.index(weight) + 1 if weight > 0 else None
        
        # Try to get registration info if available
        try:
            reg_blocks_list = metagraph.registration_blocks.tolist() if hasattr(metagraph.registration_blocks, 'tolist') else list(metagraph.registration_blocks)
            registration_block = reg_blocks_list[hotkey_index]
            age_blocks = current_block - registration_block if registration_block > 0 else 0
        except AttributeError:
            registration_block = None
            age_blocks = None
        
        status_info = {
            "uid": uid,
            "hotkey": hotkey,
            "hotkey_short": f"{hotkey[:8]}...{hotkey[-8:]}",
            "is_registered": True,
            "is_active": is_active,
            "stake": stake,
            "weight": weight,
            "rank": rank,
            "registration_block": registration_block,
            "current_block": current_block,
            "age_blocks": age_blocks,
            "commitment_url": commitment_url,
            "has_commitment": commitment_url is not None,
            "subnet_info": {
                "netuid": NETUID,
                "network": NETWORK,
                "total_miners": len(uids_list),
                "active_miners": sum(1 for s in stake_list if s > 0)
            }
        }
        
        return status_info
        
    except Exception as e:
        logger.error(f"Error getting miner status: {e}")
        return {"error": str(e)}

def print_miner_status(status: Dict[str, Any]):
    """Print miner status in a formatted way."""
    if "error" in status:
        print(f"❌ Error: {status['error']}")
        return
    
    print(f"\n🔍 Miner Status for Hotkey {status['hotkey_short']}")
    print("=" * 50)
    print(f"UID:    {status['uid']}")
    print(f"Hotkey: {status['hotkey']}")
    print(f"Status: {'🟢 Active' if status['is_active'] else '🔴 Inactive'}")
    print(f"Stake:  {status['stake']:.6f} TAO")
    print(f"Weight: {status['weight']:.6f}")
    
    if status['rank']:
        print(f"Rank:   #{status['rank']} of {status['subnet_info']['active_miners']}")
    else:
        print(f"Rank:   Unranked")
    
    if status['age_blocks'] is not None:
        print(f"Age:    {status['age_blocks']} blocks")
    else:
        print(f"Age:    Unknown")
    print(f"Commit: {'✅ Yes' if status['has_commitment'] else '❌ No'}")
    
    if status['commitment_url']:
        print(f"URL:    {status['commitment_url']}")
    
    print(f"\nSubnet Info:")
    print(f"  Network: {status['subnet_info']['network']}")
    print(f"  NetUID:  {status['subnet_info']['netuid']}")
    print(f"  Total:   {status['subnet_info']['total_miners']} miners")
    print(f"  Active:  {status['subnet_info']['active_miners']} miners")

def main():
    """Main function to check miner status."""
    print(f"Checking status for miner hotkey {TARGET_HOTKEY} on MANTIS subnet...")
    
    status = get_miner_status(TARGET_HOTKEY)
    print_miner_status(status)
    
    # Also print JSON format for programmatic use
    print(f"\n📋 JSON Output:")
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
