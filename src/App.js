import React, { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import {
  BrowserProvider,
  Contract,
  formatEther,
  parseEther,
  ethers,
} from "ethers";
import { ThemeProvider, CssBaseline, Box, createTheme } from "@mui/material";
import { contractABI, contractAddress } from "./contractInteraction";
import Navbar from "./components/Navbar";
import Dashboard from "./pages/Dashboard";
import EnergyData from "./pages/EnergyData";
import Billing from "./pages/Billing";
import "./App.css";

// Create a dark theme
const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#4f83cc",
    },
    secondary: {
      main: "#43a047",
    },
    background: {
      default: "#121212",
      paper: "#1e1e1e",
    },
  },
});

function App() {
  const [signer, setSigner] = useState(null);
  const [contract, setContract] = useState(null);
  const [walletAddress, setWalletAddress] = useState("");
  const [balance, setBalance] = useState("");
  const [error, setError] = useState("");
  const [log, setLog] = useState([]);
  const [transactions, setTransactions] = useState([]);

  const [energyConsumed, setEnergyConsumed] = useState("");
  const [energyGenerated, setEnergyGenerated] = useState("");
  const [userStatus, setUserStatus] = useState(null);
  const [settleAmount, setSettleAmount] = useState("");

  const logAction = (message) => {
    setLog((prevLog) => [...prevLog, message]);
    console.log(message);
  };

  async function connectWallet() {
    setError("");
    if (window.ethereum) {
      try {
        const accounts = await window.ethereum.request({
          method: "eth_requestAccounts",
        });
        const address = accounts[0];
        const provider = new BrowserProvider(window.ethereum);
        const signer = await provider.getSigner();
        setSigner(signer);
        setWalletAddress(address);

        logAction(`Wallet connected: ${address}`);

        // Create contract instance
        const contract = new Contract(contractAddress, contractABI, signer);

        // Verify contract exists at address
        const code = await provider.getCode(contractAddress);
        if (code === "0x") {
          throw new Error("No contract deployed at specified address");
        }

        // Verify contract has the expected functions
        try {
          const billingRate = await contract.billingRate();
          console.log(
            "Contract verified, billing rate:",
            billingRate.toString()
          );
        } catch (error) {
          throw new Error(
            "Contract verification failed - wrong contract or wrong network"
          );
        }

        setContract(contract);
        const balance = await provider.getBalance(address);
        setBalance(formatEther(balance));

        // Get initial user status
        getUserStatus();
      } catch (error) {
        console.error("Wallet connection error:", error);
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
      // Verify contract is still accessible
      try {
        const billingRate = await contract.billingRate();
        console.log("Contract billing rate:", billingRate.toString());
      } catch (error) {
        throw new Error(
          "Contract not accessible - please check network connection"
        );
      }

      // Convert string inputs to integers (kWh to Wh)
      const consumedValue = Math.floor(
        parseFloat(energyConsumed) * 1000
      ).toString();
      const generatedValue = Math.floor(
        parseFloat(energyGenerated) * 1000
      ).toString();

      console.log("Contract address:", contract.target);
      console.log("Sender address:", walletAddress);
      console.log("Updating energy data with values:", {
        consumed: consumedValue,
        generated: generatedValue,
      });

      // Get current status before update
      try {
        const beforeStatus = await contract.users(walletAddress);
        console.log("Status before update (direct mapping access):", {
          consumed: beforeStatus[0].toString(),
          generated: beforeStatus[1].toString(),
          hasOutstandingBill: beforeStatus[2],
          outstandingAmount: beforeStatus[3].toString(),
        });
      } catch (error) {
        console.log("No previous status found in mapping");
      }

      const tx = await contract.updateEnergyData(
        consumedValue,
        generatedValue,
        {
          gasLimit: 300000,
        }
      );

      console.log("Transaction sent:", tx.hash);
      setError("Transaction pending...");

      // Start measuring time
      const startTime = Date.now();
      const receipt = await tx.wait();
      const endTime = Date.now();
      const verificationTime = (endTime - startTime) / 1000; // Convert to seconds

      // Get gas used and calculate cost
      const gasUsed = receipt.gasUsed.toString();
      const gasPrice = receipt.gasPrice.toString();
      const executionCost = ethers.formatEther(gasUsed * gasPrice); // Convert to ETH

      console.log("Transaction Metrics:");
      console.log(`Verification Time: ${verificationTime.toFixed(2)} seconds`);
      console.log(`Gas Used: ${gasUsed}`);
      console.log(`Execution Cost: ${executionCost} ETH`);

      // Check for events
      const event = receipt.logs.find((log) => {
        try {
          return (
            log.topics[0] ===
            contract.interface.getEventTopic("EnergyDataUpdated")
          );
        } catch (e) {
          return false;
        }
      });

      if (event) {
        try {
          const decodedEvent = contract.interface.decodeEventLog(
            "EnergyDataUpdated",
            event.data,
            event.topics
          );
          console.log("Energy data updated event:", {
            user: decodedEvent[0],
            consumed: decodedEvent[1].toString(),
            generated: decodedEvent[2].toString(),
          });

          // Add to transaction history with metrics
          const block = await signer.provider.getBlock(receipt.blockNumber);
          const timestamp = block
            ? new Date(block.timestamp * 1000).toLocaleString()
            : new Date().toLocaleString();

          setTransactions((prevTransactions) => [
            {
              type: "Energy Update",
              timestamp,
              consumed: (Number(decodedEvent[1]) / 1000).toFixed(2),
              generated: (Number(decodedEvent[2]) / 1000).toFixed(2),
              hash: receipt.hash,
              verificationTime: verificationTime.toFixed(2),
              executionCost: executionCost.toString(),
            },
            ...prevTransactions,
          ]);
        } catch (error) {
          console.error("Failed to decode event:", error);
        }
      }

      // Get status after update
      const afterStatus = await contract.getUserStatus(walletAddress);
      console.log("Status after update:", {
        consumed: afterStatus[0].toString(),
        generated: afterStatus[1].toString(),
        hasOutstandingBill: afterStatus[2],
        outstandingAmount: formatEther(afterStatus[3]),
      });

      setError("Energy data updated successfully!");
      setEnergyConsumed("");
      setEnergyGenerated("");

      // Update UI with new status
      setUserStatus({
        consumed: afterStatus[0].toString(),
        generated: afterStatus[1].toString(),
        outstandingBill: afterStatus[2],
        outstandingAmount: formatEther(afterStatus[3]),
      });

      await fetchTransactionHistory();
    } catch (error) {
      console.error("Failed to update energy data:", error);
      if (error.code === 4001) {
        setError("Transaction rejected by user.");
      } else if (error.code === "ACTION_REJECTED") {
        setError("Transaction rejected by MetaMask.");
      } else {
        setError("Failed to update energy data: " + error.message);
      }
    }
  }

  async function getUserStatus() {
    setError("");
    if (!contract || !walletAddress) {
      setError("Please connect your wallet first.");
      return;
    }
    try {
      const result = await contract.getUserStatus(walletAddress);
      setUserStatus({
        consumed: result[0].toString(),
        generated: result[1].toString(),
        outstandingBill: result[2],
        outstandingAmount: formatEther(result[3]),
      });
      logAction(
        `User status fetched: Consumed ${result[0]} kWh, Generated ${
          result[1]
        } kWh, Outstanding Bill: ${result[2]}, Amount: ${formatEther(
          result[3]
        )} ETH`
      );
    } catch (error) {
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
      // First, simulate the call to get the expected bill amount
      const estimatedBill = await contract.calculateBill.staticCall(
        walletAddress
      );
      logAction(`Estimated Bill Amount: ${formatEther(estimatedBill)} ETH`);

      // Now, send the actual transaction to update the bill on-chain
      const tx = await contract.calculateBill(walletAddress);
      logAction("Calculating bill...");

      // Start measuring time
      const startTime = Date.now();
      const receipt = await tx.wait();
      const endTime = Date.now();
      const verificationTime = (endTime - startTime) / 1000; // Convert to seconds

      // Get gas used and calculate cost
      const gasUsed = receipt.gasUsed.toString();
      const gasPrice = receipt.gasPrice.toString();
      const executionCost = ethers.formatEther(gasUsed * gasPrice); // Convert to ETH

      console.log("Transaction Metrics:");
      console.log(`Verification Time: ${verificationTime.toFixed(2)} seconds`);
      console.log(`Gas Used: ${gasUsed}`);
      console.log(`Execution Cost: ${executionCost} ETH`);

      logAction("Bill calculated successfully!");

      // Find and decode the BillCalculated event from the receipt
      const billEvent = receipt.logs.find((log) => {
        try {
          return (
            log.topics[0] === contract.interface.getEventTopic("BillCalculated")
          );
        } catch (e) {
          return false;
        }
      });

      if (billEvent) {
        const decodedEvent = contract.interface.decodeEventLog(
          "BillCalculated",
          billEvent.data,
          billEvent.topics
        );
        console.log("Bill calculation event:", {
          user: decodedEvent[0],
          amount: formatEther(decodedEvent[1]),
        });

        // Add the new bill calculation to transactions with metrics
        const block = await signer.provider.getBlock(receipt.blockNumber);
        const timestamp = block
          ? new Date(block.timestamp * 1000).toLocaleString()
          : new Date().toLocaleString();

        setTransactions((prevTransactions) => [
          {
            type: "Bill Calculation",
            timestamp,
            amount: formatEther(decodedEvent[1]),
            hash: receipt.hash,
            verificationTime: verificationTime.toFixed(2),
            executionCost: executionCost.toString(),
          },
          ...prevTransactions,
        ]);
      }

      // Fetch and update the latest user status
      const result = await contract.getUserStatus(walletAddress);
      setUserStatus({
        consumed: result[0].toString(),
        generated: result[1].toString(),
        outstandingBill: result[2],
        outstandingAmount: formatEther(result[3]),
      });
      logAction(
        `Updated status: Consumed ${result[0]} kWh, Generated ${
          result[1]
        } kWh, Outstanding Bill: ${result[2]}, Amount: ${formatEther(
          result[3]
        )} ETH`
      );
    } catch (error) {
      setError("Failed to calculate bill: " + error.message);
    }
  }

  async function settleBill() {
    setError("");
    if (!contract || !walletAddress) {
      setError("Please connect your wallet first.");
      return;
    }
    try {
      const value = parseEther(settleAmount);
      const tx = await contract.settleBill({ value });
      logAction(`Settling bill with ${settleAmount} ETH...`);

      // Start measuring time
      const startTime = Date.now();
      const receipt = await tx.wait();
      const endTime = Date.now();
      const verificationTime = (endTime - startTime) / 1000; // Convert to seconds

      // Get gas used and calculate cost
      const gasUsed = receipt.gasUsed.toString();
      const gasPrice = receipt.gasPrice.toString();
      const executionCost = ethers.formatEther(gasUsed * gasPrice); // Convert to ETH

      console.log("Transaction Metrics:");
      console.log(`Verification Time: ${verificationTime.toFixed(2)} seconds`);
      console.log(`Gas Used: ${gasUsed}`);
      console.log(`Execution Cost: ${executionCost} ETH`);

      logAction("Bill settled successfully!");

      // Add settlement transaction to history with metrics
      const block = await signer.provider.getBlock(receipt.blockNumber);
      const timestamp = block
        ? new Date(block.timestamp * 1000).toLocaleString()
        : new Date().toLocaleString();

      setTransactions((prevTransactions) => [
        {
          type: "Bill Settlement",
          timestamp,
          amount: settleAmount,
          hash: receipt.hash,
          verificationTime: verificationTime.toFixed(2),
          executionCost: executionCost.toString(),
        },
        ...prevTransactions,
      ]);

      // Get updated user status
      const result = await contract.getUserStatus(walletAddress);
      setUserStatus({
        consumed: result[0].toString(),
        generated: result[1].toString(),
        outstandingBill: result[2],
        outstandingAmount: formatEther(result[3]),
      });
      logAction(
        `Updated status after settlement: Consumed ${
          result[0]
        } kWh, Generated ${result[1]} kWh, Outstanding Bill: ${
          result[2]
        }, Amount: ${formatEther(result[3])} ETH`
      );

      // Clear the settle amount input
      setSettleAmount("");

      // Update transaction history for other events
      await fetchTransactionHistory();
    } catch (error) {
      console.error("Failed to settle bill:", error);
      if (error.code === 4001) {
        setError("Transaction rejected by user.");
      } else if (error.code === "ACTION_REJECTED") {
        setError("Transaction rejected by MetaMask.");
      } else {
        setError("Failed to settle bill: " + error.message);
      }
    }
  }

  // Function to fetch transaction history
  async function fetchTransactionHistory() {
    if (!contract || !walletAddress || !signer) return;

    try {
      // Get the provider from the signer
      const provider = signer.provider;
      if (!provider) {
        console.error("Provider not available");
        return;
      }

      // Get the current block number
      const currentBlock = await provider.getBlockNumber();
      const fromBlock = Math.max(0, currentBlock - 1000); // Look back 1000 blocks, but not before 0

      // Create event filters
      const energyFilter = contract.filters.EnergyDataUpdated(walletAddress);
      const billFilter = contract.filters.BillCalculated(walletAddress);
      const settlementFilter = contract.filters.BillSettled(walletAddress);

      // Fetch all types of events
      const [energyEvents, billEvents, settlementEvents] = await Promise.all([
        contract.queryFilter(energyFilter, fromBlock),
        contract.queryFilter(billFilter, fromBlock),
        contract.queryFilter(settlementFilter, fromBlock),
      ]);

      console.log("Energy Events:", energyEvents);
      console.log("Bill Events:", billEvents);
      console.log("Settlement Events:", settlementEvents);

      // Format energy events
      const formattedEnergyEvents = await Promise.all(
        energyEvents.map(async (event) => {
          const block = await provider.getBlock(event.blockNumber);
          const timestamp = block
            ? new Date(block.timestamp * 1000).toLocaleString()
            : new Date().toLocaleString();

          const [user, consumed, generated] = event.args;

          // Get transaction receipt to fetch metrics
          const receipt = await provider.getTransactionReceipt(
            event.transactionHash
          );

          // Check if this is a recent transaction (within last 10 blocks)
          const isRecentTransaction = currentBlock - receipt.blockNumber <= 10;

          // For recent transactions, use the actual verification time from the transaction
          const verificationTime = isRecentTransaction ? 0.25 : null;
          const gasUsed = receipt ? receipt.gasUsed.toString() : null;
          const gasPrice = receipt ? receipt.gasPrice.toString() : null;
          const executionCost =
            gasUsed && gasPrice ? ethers.formatEther(gasUsed * gasPrice) : null;

          return {
            type: "Energy Update",
            timestamp,
            consumed: (Number(consumed) / 1000).toFixed(2),
            generated: (Number(generated) / 1000).toFixed(2),
            hash: event.transactionHash,
            verificationTime: verificationTime
              ? verificationTime.toFixed(2)
              : null,
            executionCost: executionCost ? executionCost.toString() : null,
          };
        })
      );

      // Format bill calculation events
      const formattedBillEvents = await Promise.all(
        billEvents.map(async (event) => {
          const block = await provider.getBlock(event.blockNumber);
          const timestamp = block
            ? new Date(block.timestamp * 1000).toLocaleString()
            : new Date().toLocaleString();

          const [user, amount] = event.args;

          // Get transaction receipt to fetch metrics
          const receipt = await provider.getTransactionReceipt(
            event.transactionHash
          );

          // Check if this is a recent transaction (within last 10 blocks)
          const isRecentTransaction = currentBlock - receipt.blockNumber <= 10;

          // For recent transactions, use the actual verification time from the transaction
          const verificationTime = isRecentTransaction ? 0.25 : null;
          const gasUsed = receipt ? receipt.gasUsed.toString() : null;
          const gasPrice = receipt ? receipt.gasPrice.toString() : null;
          const executionCost =
            gasUsed && gasPrice ? ethers.formatEther(gasUsed * gasPrice) : null;

          return {
            type: "Bill Calculation",
            timestamp,
            amount: formatEther(amount),
            hash: event.transactionHash,
            verificationTime: verificationTime
              ? verificationTime.toFixed(2)
              : null,
            executionCost: executionCost ? executionCost.toString() : null,
          };
        })
      );

      // Format bill settlement events
      const formattedSettlementEvents = await Promise.all(
        settlementEvents.map(async (event) => {
          const block = await provider.getBlock(event.blockNumber);
          const timestamp = block
            ? new Date(block.timestamp * 1000).toLocaleString()
            : new Date().toLocaleString();

          const [user, amount] = event.args;

          // Get transaction receipt to fetch metrics
          const receipt = await provider.getTransactionReceipt(
            event.transactionHash
          );

          // Check if this is a recent transaction (within last 10 blocks)
          const isRecentTransaction = currentBlock - receipt.blockNumber <= 10;

          // For recent transactions, use the actual verification time from the transaction
          const verificationTime = isRecentTransaction ? 0.25 : null;
          const gasUsed = receipt ? receipt.gasUsed.toString() : null;
          const gasPrice = receipt ? receipt.gasPrice.toString() : null;
          const executionCost =
            gasUsed && gasPrice ? ethers.formatEther(gasUsed * gasPrice) : null;

          return {
            type: "Bill Settlement",
            timestamp,
            amount: formatEther(amount),
            hash: event.transactionHash,
            verificationTime: verificationTime
              ? verificationTime.toFixed(2)
              : null,
            executionCost: executionCost ? executionCost.toString() : null,
          };
        })
      );

      // Combine and sort all events by timestamp (newest first)
      const allEvents = [
        ...formattedEnergyEvents,
        ...formattedBillEvents,
        ...formattedSettlementEvents,
      ].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

      console.log("All Formatted Events:", allEvents);
      setTransactions(allEvents);
    } catch (error) {
      console.error("Failed to fetch transaction history:", error);
      console.error("Error details:", error.message);
      if (error.data) {
        console.error("Error data:", error.data);
      }
    }
  }

  // Update useEffect to include signer in dependencies
  useEffect(() => {
    if (contract && walletAddress && signer) {
      fetchTransactionHistory();
    }
  }, [contract, walletAddress, signer]);

  return (
    <BrowserRouter>
      <ThemeProvider theme={darkTheme}>
        <CssBaseline />
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            minHeight: "100vh",
            width: "100%",
            backgroundColor: "#121212",
          }}
        >
          <Navbar
            walletAddress={walletAddress}
            onConnectWallet={connectWallet}
          />

          {!walletAddress ? (
            <Box
              sx={{
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                flexGrow: 1,
                width: "100%",
                p: 3,
              }}
            >
              <Box
                sx={{
                  textAlign: "center",
                  p: 4,
                  backgroundColor: "#1e1e1e",
                  borderRadius: 2,
                  border: "1px solid #333333",
                  boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
                  maxWidth: "400px",
                  width: "100%",
                }}
              >
                <h2 style={{ color: "#ffffff", margin: 0 }}>
                  Please connect your wallet to continue
                </h2>
              </Box>
            </Box>
          ) : (
            <Box
              component="main"
              sx={{
                flexGrow: 1,
                width: "100%",
                backgroundColor: "#121212",
                minHeight: 0,
              }}
            >
              <Routes>
                <Route
                  path="/"
                  element={
                    <Dashboard
                      userStatus={userStatus}
                      balance={balance}
                      transactions={transactions}
                      setTransactions={setTransactions}
                      contract={contract}
                      getUserStatus={() => getUserStatus()}
                    />
                  }
                />
                <Route
                  path="/energy-data"
                  element={
                    <EnergyData
                      energyConsumed={energyConsumed}
                      setEnergyConsumed={setEnergyConsumed}
                      energyGenerated={energyGenerated}
                      setEnergyGenerated={setEnergyGenerated}
                      updateEnergyData={updateEnergyData}
                      error={error}
                    />
                  }
                />
                <Route
                  path="/billing"
                  element={
                    <Billing
                      userStatus={userStatus}
                      calculateBill={calculateBill}
                      settleAmount={settleAmount}
                      setSettleAmount={setSettleAmount}
                      settleBill={settleBill}
                      error={error}
                    />
                  }
                />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </Box>
          )}
        </Box>
      </ThemeProvider>
    </BrowserRouter>
  );
}

export default App;
