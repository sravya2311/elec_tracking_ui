# Contract ABI and address
contractABI = [
    {
        "inputs": [
            {"internalType": "string", "name": "_consumed", "type": "string"},
            {"internalType": "string", "name": "_generated", "type": "string"}
        ],
        "name": "updateEnergyData",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "user", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "consumed", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "generated", "type": "uint256"}
        ],
        "name": "EnergyDataUpdated",
        "type": "event"
    }
]

contractAddress = "0x5FbDB2315678afecb367f032d93F642f64180aa3"  # Replace with your deployed contract address 