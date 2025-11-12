import json, os
from web3 import Web3
from django.conf import settings

ABI_PATH = os.path.join(os.path.dirname(__file__), "contract_abi.json")
ADDR_PATH = os.path.join(os.path.dirname(__file__), "contract_address.txt")

def _web3():
    return Web3(Web3.HTTPProvider(settings.RPC_URL))

def _account(w3: Web3):
    return w3.eth.account.from_key(settings.PRIVATE_KEY)

def load_contract():
    if not os.path.exists(ABI_PATH) or not os.path.exists(ADDR_PATH):
        return None, None, None
    with open(ABI_PATH, "r") as f:
        abi = json.load(f)
    with open(ADDR_PATH, "r") as f:
        address = f.read().strip()
    w3 = _web3()
    contract = w3.eth.contract(address=Web3.to_checksum_address(address), abi=abi)
    account = _account(w3)
    return w3, account, contract

def record_cid(s_code: str, cid: str):
    w3, acct, contract = load_contract()
    if contract is None:
        return None
    nonce = w3.eth.get_transaction_count(acct.address)
    tx = contract.functions.recordPaper(s_code, cid).build_transaction({
        "from": acct.address,
        "nonce": nonce,
        "gas": 1_500_000,
        "gasPrice": w3.to_wei("1", "gwei"),
    })
    signed = w3.eth.account.sign_transaction(tx, private_key=settings.PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt.transactionHash.hex()
