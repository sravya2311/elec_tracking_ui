import React from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Alert,
  Grid,
} from "@mui/material";
import AccountBalanceWalletIcon from "@mui/icons-material/AccountBalanceWallet";
import CalculateIcon from "@mui/icons-material/Calculate";
import PaymentsIcon from "@mui/icons-material/Payments";

const Billing = ({
  userStatus,
  calculateBill,
  settleAmount,
  setSettleAmount,
  settleBill,
  error,
}) => {
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ color: "#1a237e", mb: 4 }}>
        Billing Management
      </Typography>

      {error && (
        <Alert
          severity={error.includes("success") ? "success" : "error"}
          sx={{ mb: 3 }}
        >
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card
            sx={{
              height: "100%",
              backgroundColor: "#f5f5f5",
              boxShadow: 3,
            }}
          >
            <CardContent>
              <Box sx={{ display: "flex", alignItems: "center", mb: 3 }}>
                <CalculateIcon sx={{ color: "#1565c0", mr: 2, fontSize: 30 }} />
                <Typography variant="h5">Calculate Bill</Typography>
              </Box>

              <Box sx={{ mb: 3 }}>
                <Typography variant="body1" gutterBottom>
                  Current Status:
                </Typography>
                <Typography variant="h6" color="primary">
                  Outstanding Amount: {userStatus?.outstandingAmount} ETH
                </Typography>
              </Box>

              <Button
                variant="contained"
                onClick={calculateBill}
                fullWidth
                sx={{
                  backgroundColor: "#1565c0",
                  "&:hover": {
                    backgroundColor: "#1976d2",
                  },
                  py: 1.5,
                }}
              >
                Calculate Current Bill
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card
            sx={{
              height: "100%",
              backgroundColor: "#f5f5f5",
              boxShadow: 3,
            }}
          >
            <CardContent>
              <Box sx={{ display: "flex", alignItems: "center", mb: 3 }}>
                <PaymentsIcon sx={{ color: "#2e7d32", mr: 2, fontSize: 30 }} />
                <Typography variant="h5">Settle Bill</Typography>
              </Box>

              <TextField
                label="Amount to Pay (ETH)"
                type="number"
                value={settleAmount}
                onChange={(e) => setSettleAmount(e.target.value)}
                fullWidth
                InputProps={{
                  inputProps: { min: 0, step: "0.01" },
                }}
                sx={{
                  mb: 3,
                  "& .MuiOutlinedInput-root": {
                    backgroundColor: "#ffffff",
                  },
                }}
              />

              <Button
                variant="contained"
                onClick={settleBill}
                fullWidth
                sx={{
                  backgroundColor: "#2e7d32",
                  "&:hover": {
                    backgroundColor: "#388e3c",
                  },
                  py: 1.5,
                }}
                startIcon={<AccountBalanceWalletIcon />}
              >
                Pay Bill
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Billing;
