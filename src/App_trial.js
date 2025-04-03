import React, { useState } from "react";
import { BrowserProvider, Contract, formatEther, parseEther } from "ethers";
import { contractABI, contractAddress } from "./contractInteraction";
import "./App.css";

function App() {
  const [signer, setSigner] = useState(null);
  const [contract, setContract] = useState(null);
  const [walletAddress, setWalletAddress] = useState("");
  const [balance, setBalance] = useState("");
  const [contractAddressDisplay, setContractAddressDisplay] = useState("");
  const [error, setError] = useState("");

  const [energyConsumed, setEnergyConsumed] = useState("");
  const [energyGenerated, setEnergyGenerated] = useState("");
  const [userStatus, setUserStatus] = useState(null);
  const [settleAmount, setSettleAmount] = useState("");

  // Switch to correct network (e.g., Hardhat or Rinkeby)
  const switchToCorrectNetwork = async () => {
    const chainId = "0x7A69"; // Hardhat network chain ID in hexadecimal
    try {
      await window.ethereum.request({
        method: "wallet_switchEthereumChain",
        params: [{ chainId }],
      });
    } catch (error) {
      if (error.code === 4902) {
        setError("Network not found. Please add Hardhat network to MetaMask.");
      } else {
        setError("Failed to switch network: " + error.message);
      }
      throw error;
    }
  };

  async function connectWallet() {
    setError("");
    if (window.ethereum) {
      try {
        await switchToCorrectNetwork();

        const accounts = await window.ethereum.request({
          method: "eth_requestAccounts",
        });

        const address = accounts[0];
        const provider = new BrowserProvider(window.ethereum);
        const signer = await provider.getSigner();
        setSigner(signer);
        setWalletAddress(address);

        console.log("Wallet connected:", address);

        const contract = new Contract(contractAddress, contractABI, signer);
        setContract(contract);
        setContractAddressDisplay(contractAddress);

        const balance = await provider.getBalance(address);
        setBalance(formatEther(balance));
      } catch (error) {
        console.error("Failed to connect wallet or fetch balance:", error);
        setError("Failed to connect wallet: " + error.message);
      }
    } else {
      setError("Ethereum provider not found. Please install MetaMask.");
    }
  }

  async function updateEnergyData() {
    setError("");
    if (!contract || !walletAddress) {
      setError("Please connect your wallet first.");
      return;
    }

    if (!energyConsumed || !energyGenerated) {
      setError("Please enter both energy consumed and generated values.");
      return;
    }

    try {
      const consumedValue = Math.floor(parseFloat(energyConsumed)).toString();
      const generatedValue = Math.floor(parseFloat(energyGenerated)).toString();

      const tx = await contract.updateEnergyData(
        walletAddress,
        consumedValue,
        generatedValue,
        {
          gasLimit: 3000000, // Adjust gas limit if needed
        }
      );

      setError("Transaction pending...");
      await tx.wait();
      setError("Energy data updated successfully!");

      setEnergyConsumed("");
      setEnergyGenerated("");
    } catch (error) {
      console.error("Failed to update energy data:", error);
      handleTransactionError(error);
    }
  }

  async function getUserStatus() {
    setError("");
    if (!contract || !walletAddress) {
      setError("Please connect your wallet first.");
      return;
    }

    try {
      console.log("Fetching status for:", walletAddress);

      const result = await contract.getUserStatus(walletAddress);
      console.log("Raw result:", result);

      if (!result || result.length < 4) {
        throw new Error("Invalid response from contract.");
      }

      setUserStatus({
        consumed: result[0].toString(),
        generated: result[1].toString(),
        outstandingBill: result[2],
        outstandingAmount: formatEther(result[3]),
      });

      console.log("User status updated:", userStatus);
    } catch (error) {
      console.error("Failed to fetch user status:", error);
      setError("Failed to fetch user status: " + error.message);
    }
  }

  async function calculateBill() {
    setError("");
    if (!contract || !walletAddress) {
      setError("Please connect your wallet first.");
      return;
    }

    try {
      const tx = await contract.calculateBill(walletAddress);

      setError("Calculating bill...");
      await tx.wait();
      setError("Bill calculated successfully!");
    } catch (error) {
      console.error("Failed to calculate bill:", error);
      handleTransactionError(error);
    }
  }

  async function settleBill() {
    setError("");
    if (!contract || !walletAddress) {
      setError("Please connect your wallet first.");
      return;
    }

    if (!settleAmount || parseFloat(settleAmount) <= 0) {
      setError("Please enter a valid amount to settle.");
      return;
    }

    try {
      const value = parseEther(settleAmount);
      const tx = await contract.settleBill({
        value,
        gasLimit: 300000,
      });

      setError("Settling bill...");
      await tx.wait();
      setError("Bill settled successfully!");
      setSettleAmount("");
    } catch (error) {
      console.error("Failed to settle bill:", error);
      handleTransactionError(error);
    }
  }

  function handleTransactionError(error) {
    if (error.code === 4100) {
      setError("Transaction not authorized. Please approve in MetaMask.");
    } else if (error.code === "ACTION_REJECTED") {
      setError("Transaction rejected by user.");
    } else {
      setError("Transaction failed: " + error.message);
    }
  }

  function disconnectWallet() {
    setSigner(null);
    setWalletAddress("");
    setBalance("");
    setContractAddressDisplay("");
    setUserStatus(null);
    setEnergyConsumed("");
    setEnergyGenerated("");
    setSettleAmount("");
    setError("");
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>Energy Usage Tracker</h1>
        {error && (
          <div
            className="error-message"
            style={{
              color: error.includes("success") ? "green" : "red",
              margin: "10px",
            }}
          >
            {error}
          </div>
        )}
        {walletAddress ? (
          <>
            <div className="functions-container">
              {/* Wallet Details and Functions */}
              <div className="function-box">
                <p>Wallet Address: {walletAddress}</p>
                <p>Balance: {balance} ETH</p>
                <p>Contract Address: {contractAddressDisplay}</p>
              </div>
              <div className="function-box">
                <h3>Update Energy Data</h3>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  placeholder="Energy Consumed"
                  value={energyConsumed}
                  onChange={(e) => setEnergyConsumed(e.target.value)}
                />
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  placeholder="Energy Generated"
                  value={energyGenerated}
                  onChange={(e) => setEnergyGenerated(e.target.value)}
                />
                <button onClick={updateEnergyData}>Update Energy Data</button>
              </div>

              <div className="function-box">
                <h3>Calculate Bill</h3>
                <button onClick={calculateBill}>Calculate Bill</button>
              </div>

              <div className="function-box">
                <h3>Get User Status</h3>
                <button onClick={getUserStatus}>Get Status</button>
                {userStatus && (
                  <div>
                    <p>Energy Consumed: {userStatus.consumed} kWh</p>
                    <p>Energy Generated: {userStatus.generated} kWh</p>
                    <p>
                      Outstanding Bill:{" "}
                      {userStatus.outstandingBill ? "Yes" : "No"}
                    </p>
                    <p>
                      Outstanding Amount: {userStatus.outstandingAmount} ETH
                    </p>
                  </div>
                )}
              </div>

              <div className="function-box">
                <h3>Settle Bill</h3>
                <input
                  type="number"
                  step="0.0001"
                  min="0"
                  placeholder="Amount to Pay (ETH)"
                  value={settleAmount}
                  onChange={(e) => setSettleAmount(e.target.value)}
                />
                <button onClick={settleBill}>Settle Bill</button>
              </div>
              <button onClick={disconnectWallet}>Disconnect Wallet</button>
            </div>
          </>
        ) : (
          <button onClick={connectWallet}>Connect Wallet</button>
        )}
      </header>
    </div>
  );
}

export default App;
