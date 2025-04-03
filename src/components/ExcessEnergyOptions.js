import React, { useState } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Box,
  Typography,
  Alert,
} from "@mui/material";

const ExcessEnergyOptions = ({
  open,
  onClose,
  excessEnergy,
  onAuctionEnergy,
}) => {
  const [amount, setAmount] = useState("");
  const [serviceType, setServiceType] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = () => {
    if (!amount || !serviceType) {
      setError("Please fill in all fields");
      return;
    }

    const numAmount = parseFloat(amount);
    if (isNaN(numAmount) || numAmount <= 0) {
      setError("Please enter a valid amount");
      return;
    }

    if (numAmount > parseFloat(excessEnergy)) {
      setError("Amount cannot exceed available excess energy");
      return;
    }

    onAuctionEnergy(numAmount, serviceType);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Manage Excess Energy</DialogTitle>
      <DialogContent>
        <Box sx={{ mb: 3 }}>
          <Typography variant="body1" gutterBottom>
            Available Excess Energy: {excessEnergy} kWh
          </Typography>
          <Alert severity="info" sx={{ mt: 2 }}>
            You can auction your excess energy to public services. The highest
            bidder will receive the energy.
          </Alert>
        </Box>

        <TextField
          fullWidth
          label="Amount (kWh)"
          value={amount}
          onChange={(e) => setAmount(e.target.value)}
          type="number"
          sx={{ mb: 2 }}
        />

        <FormControl fullWidth sx={{ mb: 2 }}>
          <InputLabel>Service Type</InputLabel>
          <Select
            value={serviceType}
            onChange={(e) => setServiceType(e.target.value)}
            label="Service Type"
          >
            <MenuItem value="Police">Police Station</MenuItem>
            <MenuItem value="Hospital">Hospital</MenuItem>
            <MenuItem value="Fire">Fire Department</MenuItem>
            <MenuItem value="Emergency">Emergency Services</MenuItem>
          </Select>
        </FormControl>

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleSubmit} variant="contained" color="primary">
          Start Auction
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ExcessEnergyOptions;
