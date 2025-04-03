import React, { useState } from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Link,
  Chip,
  Button,
  Alert,
} from "@mui/material";
import BoltIcon from "@mui/icons-material/Bolt";
import SolarPowerIcon from "@mui/icons-material/SolarPower";
import AccountBalanceIcon from "@mui/icons-material/AccountBalance";
import ReceiptLongIcon from "@mui/icons-material/ReceiptLong";
import UpdateIcon from "@mui/icons-material/Update";
import AssessmentIcon from "@mui/icons-material/Assessment";
import TrendingUpIcon from "@mui/icons-material/TrendingUp";
import ExcessEnergyOptions from "../components/ExcessEnergyOptions";
import { ethers } from "ethers";

const Dashboard = ({
  userStatus,
  balance,
  transactions,
  setTransactions,
  contract,
  getUserStatus,
}) => {
  const [excessEnergyDialogOpen, setExcessEnergyDialogOpen] = useState(false);
  const [error, setError] = useState("");
  const truncateHash = (hash) =>
    hash ? `${hash.slice(0, 6)}...${hash.slice(-4)}` : "N/A";

  // Calculate excess energy
  const excessEnergy = userStatus
    ? Math.max(
        0,
        parseFloat(userStatus.generated) - parseFloat(userStatus.consumed)
      ).toFixed(2)
    : 0;

  const handleViewStats = async () => {
    try {
      const response = await fetch("http://localhost:5000/start-streamlit", {
        method: "POST",
      });

      if (response.ok) {
        const data = await response.json();
        window.open("http://localhost:8501", "_blank");
      } else {
        console.error("Failed to start Streamlit server");
      }
    } catch (error) {
      console.error("Error starting Streamlit server:", error);
    }
  };

  const handleAuctionEnergy = async (amount, serviceType) => {
    try {
      setError("");
      if (!contract) {
        setError("Contract not initialized");
        return;
      }

      // Get the signer from the contract's runner provider
      const signer = await contract.runner.provider.getSigner();
      const signerAddress = await signer.getAddress();

      // Get current user status
      const userStatus = await contract.getUserStatus(signerAddress);
      const excessEnergy =
        (Number(userStatus[1]) - Number(userStatus[0])) / 1000; // Convert to kWh

      if (amount > excessEnergy) {
        setError("Cannot auction more energy than available excess");
        return;
      }

      // Convert kWh to Wh for the contract
      const amountInWh = Math.floor(amount * 1000).toString();

      // Start the auction and measure time
      const startTime = Date.now();
      const tx = await contract.startEnergyAuction(amountInWh, serviceType, {
        gasLimit: 300000,
      });

      setError("Transaction pending...");

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

      // Add to transaction history with metrics
      const block = await signer.provider.getBlock(receipt.blockNumber);
      const timestamp = block
        ? new Date(block.timestamp * 1000).toLocaleString()
        : new Date().toLocaleString();

      setTransactions((prevTransactions) => [
        {
          type: "Energy Auction",
          timestamp,
          amount: amount,
          serviceType: serviceType,
          hash: receipt.hash,
          verificationTime: verificationTime.toFixed(2),
          executionCost: executionCost.toString(),
        },
        ...prevTransactions,
      ]);

      setError("Auction started successfully!");
      setExcessEnergyDialogOpen(false);

      // Refresh user status to update the excess energy display
      await getUserStatus();
    } catch (error) {
      console.error("Failed to start auction:", error);
      if (error.code === 4001) {
        setError("Transaction rejected by user.");
      } else if (error.code === "ACTION_REJECTED") {
        setError("Transaction rejected by MetaMask.");
      } else {
        setError("Failed to start auction: " + error.message);
      }
    }
  };

  return (
    <Box sx={{ p: 4 }}>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          mb: 4,
        }}
      >
        <Typography
          variant="h4"
          sx={{
            background: "linear-gradient(45deg, #4f83cc, #60a5fa)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            fontWeight: 600,
          }}
        >
          Dashboard Overview
        </Typography>
        <Button
          variant="contained"
          startIcon={<AssessmentIcon />}
          onClick={handleViewStats}
          sx={{
            background: "linear-gradient(45deg, #43a047, #66bb6a)",
            color: "white",
            "&:hover": {
              background: "linear-gradient(45deg, #2e7d32, #43a047)",
            },
            boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
            borderRadius: "8px",
            padding: "10px 20px",
          }}
        >
          View Stats & Patterns
        </Button>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={3}>
          <Card
            className="dashboard-card-consumed"
            sx={{
              height: "100%",
              transition: "transform 0.2s, box-shadow 0.2s",
              "&:hover": {
                transform: "scale(1.02)",
                boxShadow: "0 8px 16px rgba(0, 0, 0, 0.2)",
              },
            }}
          >
            <CardContent>
              <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                <BoltIcon sx={{ color: "#f57c00", mr: 1, fontSize: 30 }} />
                <Typography variant="h6" sx={{ color: "#f57c00" }}>
                  Energy Consumed
                </Typography>
              </Box>
              {userStatus ? (
                <Typography variant="h4" sx={{ color: "#ffffff" }}>
                  {userStatus.consumed} kWh
                </Typography>
              ) : (
                <CircularProgress size={24} sx={{ color: "#f57c00" }} />
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card
            className="dashboard-card-generated"
            sx={{
              height: "100%",
              transition: "transform 0.2s, box-shadow 0.2s",
              "&:hover": {
                transform: "scale(1.02)",
                boxShadow: "0 8px 16px rgba(0, 0, 0, 0.2)",
              },
            }}
          >
            <CardContent>
              <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                <SolarPowerIcon
                  sx={{ color: "#43a047", mr: 1, fontSize: 30 }}
                />
                <Typography variant="h6" sx={{ color: "#43a047" }}>
                  Energy Generated
                </Typography>
              </Box>
              {userStatus ? (
                <Typography variant="h4" sx={{ color: "#ffffff" }}>
                  {userStatus.generated} kWh
                </Typography>
              ) : (
                <CircularProgress size={24} sx={{ color: "#43a047" }} />
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card
            className="dashboard-card-excess"
            sx={{
              height: "100%",
              transition: "transform 0.2s, box-shadow 0.2s",
              "&:hover": {
                transform: "scale(1.02)",
                boxShadow: "0 8px 16px rgba(0, 0, 0, 0.2)",
              },
              cursor: parseFloat(excessEnergy) > 0 ? "pointer" : "default",
            }}
            onClick={() =>
              parseFloat(excessEnergy) > 0 && setExcessEnergyDialogOpen(true)
            }
          >
            <CardContent>
              <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                <TrendingUpIcon
                  sx={{ color: "#9c27b0", mr: 1, fontSize: 30 }}
                />
                <Typography variant="h6" sx={{ color: "#9c27b0" }}>
                  Excess Energy
                </Typography>
              </Box>
              {userStatus ? (
                <>
                  <Typography
                    variant="h4"
                    sx={{
                      color:
                        parseFloat(excessEnergy) >= 0 ? "#9c27b0" : "#f44336",
                    }}
                  >
                    {excessEnergy} kWh
                  </Typography>
                  {parseFloat(excessEnergy) > 0 && (
                    <Typography
                      variant="body2"
                      sx={{ mt: 1, color: "#9c27b0" }}
                    >
                      Click to manage excess energy
                    </Typography>
                  )}
                </>
              ) : (
                <CircularProgress size={24} sx={{ color: "#9c27b0" }} />
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card
            className="dashboard-card-balance"
            sx={{
              height: "100%",
              transition: "transform 0.2s, box-shadow 0.2s",
              "&:hover": {
                transform: "scale(1.02)",
                boxShadow: "0 8px 16px rgba(0, 0, 0, 0.2)",
              },
            }}
          >
            <CardContent>
              <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                <AccountBalanceIcon
                  sx={{ color: "#4f83cc", mr: 1, fontSize: 30 }}
                />
                <Typography variant="h6" sx={{ color: "#4f83cc" }}>
                  Outstanding Amount
                </Typography>
              </Box>
              {userStatus ? (
                <Typography variant="h4" sx={{ color: "#ffffff" }}>
                  {userStatus.outstandingAmount} ETH
                </Typography>
              ) : (
                <CircularProgress size={24} sx={{ color: "#4f83cc" }} />
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Box sx={{ mt: 2, mb: 4 }}>
            <Typography
              variant="h5"
              gutterBottom
              sx={{
                color: "#4f83cc",
                fontWeight: 600,
              }}
            >
              Wallet Balance
            </Typography>
            <Typography
              variant="h3"
              sx={{
                background: "linear-gradient(45deg, #4f83cc, #60a5fa)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                fontWeight: 600,
              }}
            >
              {balance} ETH
            </Typography>
          </Box>
        </Grid>

        <Grid item xs={12}>
          <Card
            sx={{
              backgroundColor: "#1e1e1e",
              border: "1px solid #333333",
            }}
          >
            <CardContent>
              <Box sx={{ display: "flex", alignItems: "center", mb: 3 }}>
                <ReceiptLongIcon
                  sx={{ color: "#4f83cc", mr: 2, fontSize: 30 }}
                />
                <Typography variant="h5" sx={{ color: "#ffffff" }}>
                  Transaction History
                </Typography>
              </Box>

              <TableContainer
                component={Paper}
                sx={{ backgroundColor: "transparent" }}
              >
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Type</TableCell>
                      <TableCell>Timestamp</TableCell>
                      <TableCell>Details</TableCell>
                      <TableCell>Execution Cost</TableCell>
                      <TableCell>Transaction Hash</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {transactions.map((tx, index) => (
                      <TableRow key={index}>
                        <TableCell>
                          <Chip
                            icon={
                              tx.type === "Energy Update" ? (
                                <UpdateIcon />
                              ) : (
                                <AccountBalanceIcon />
                              )
                            }
                            label={tx.type}
                            sx={{
                              backgroundColor:
                                tx.type === "Energy Update"
                                  ? "rgba(67, 160, 71, 0.1)"
                                  : "rgba(79, 131, 204, 0.1)",
                              color:
                                tx.type === "Energy Update"
                                  ? "#43a047"
                                  : "#4f83cc",
                            }}
                          />
                        </TableCell>
                        <TableCell>{tx.timestamp}</TableCell>
                        <TableCell>
                          {tx.type === "Energy Update"
                            ? `Consumed: ${tx.consumed} kWh, Generated: ${tx.generated} kWh`
                            : `Amount: ${tx.amount} kWh, Service: ${tx.serviceType}`}
                        </TableCell>
                        <TableCell>
                          {tx.executionCost ? `${tx.executionCost} ETH` : "N/A"}
                        </TableCell>
                        <TableCell>
                          <Link
                            href={`https://sepolia.etherscan.io/tx/${tx.hash}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            sx={{ color: "#4f83cc" }}
                          >
                            {truncateHash(tx.hash)}
                          </Link>
                        </TableCell>
                      </TableRow>
                    ))}
                    {transactions.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={4} align="center">
                          No transactions found
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <ExcessEnergyOptions
        open={excessEnergyDialogOpen}
        onClose={() => setExcessEnergyDialogOpen(false)}
        excessEnergy={excessEnergy}
        onAuctionEnergy={handleAuctionEnergy}
      />
    </Box>
  );
};

export default Dashboard;
