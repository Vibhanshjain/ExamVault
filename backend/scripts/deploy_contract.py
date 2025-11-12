import json, os
from solcx import compile_standard, install_solc, set_solc_version
from web3 import Web3
from dotenv import load_dotenv

# Base path (backend/)
BASE = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE, ".env"))

RPC_URL = os.getenv("RPC_URL")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

# Connect to blockchain
w3 = Web3(Web3.HTTPProvider(RPC_URL))
account = w3.eth.account.from_key(PRIVATE_KEY)

# Compile Solidity contract
with open(os.path.join(BASE, "exams", "contracts", "ExamPapers.sol"), "r") as f:
    source = f.read()

install_solc("0.8.0")
set_solc_version("0.8.0")
compiled = compile_standard({
    "language": "Solidity",
    "sources": {"ExamPapers.sol": {"content": source}},
    "settings": {"outputSelection": {"*": {"*": ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]}}}
})

abi = compiled["contracts"]["ExamPapers.sol"]["ExamPapers"]["abi"]
bytecode = compiled["contracts"]["ExamPapers.sol"]["ExamPapers"]["evm"]["bytecode"]["object"]

# Deploy contract
ExamPapers = w3.eth.contract(abi=abi, bytecode=bytecode)
nonce = w3.eth.get_transaction_count(account.address)
tx = ExamPapers.constructor().build_transaction({
    "from": account.address,
    "nonce": nonce,
    "gas": 3_000_000,
    "gasPrice": w3.to_wei("1", "gwei"),
})

signed = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)  # FIXED here
receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

contract_address = receipt.contractAddress
print("Deployed at:", contract_address)

# Save ABI & address for Django
ABI_PATH = os.path.join(BASE, "exams", "contract_abi.json")
ADDR_PATH = os.path.join(BASE, "exams", "contract_address.txt")
with open(ABI_PATH, "w") as f:
    json.dump(abi, f)
with open(ADDR_PATH, "w") as f:
    f.write(contract_address)

print("Saved ABI and address.")
