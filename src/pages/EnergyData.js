import React from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Alert,
} from "@mui/material";
import ElectricBoltIcon from "@mui/icons-material/ElectricBolt";

const EnergyData = ({
  energyConsumed,
  setEnergyConsumed,
  energyGenerated,
  setEnergyGenerated,
  updateEnergyData,
  error,
}) => {
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ color: "#1a237e", mb: 4 }}>
        Update Energy Data
      </Typography>

      <Card
        sx={{
          maxWidth: 600,
          mx: "auto",
          backgroundColor: "#f5f5f5",
          boxShadow: 3,
        }}
      >
        <CardContent>
          <Box sx={{ display: "flex", alignItems: "center", mb: 3 }}>
            <ElectricBoltIcon sx={{ color: "#f57c00", mr: 2, fontSize: 30 }} />
            <Typography variant="h5">Energy Consumption Details</Typography>
          </Box>

          {error && (
            <Alert
              severity={error.includes("success") ? "success" : "error"}
              sx={{ mb: 3 }}
            >
              {error}
            </Alert>
          )}

          <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
            <TextField
              label="Energy Consumed (kWh)"
              type="number"
              value={energyConsumed}
              onChange={(e) => setEnergyConsumed(e.target.value)}
              fullWidth
              InputProps={{
                inputProps: { min: 0 },
              }}
              sx={{
                "& .MuiOutlinedInput-root": {
                  backgroundColor: "#ffffff",
                },
              }}
            />

            <TextField
              label="Energy Generated (kWh)"
              type="number"
              value={energyGenerated}
              onChange={(e) => setEnergyGenerated(e.target.value)}
              fullWidth
              InputProps={{
                inputProps: { min: 0 },
              }}
              sx={{
                "& .MuiOutlinedInput-root": {
                  backgroundColor: "#ffffff",
                },
              }}
            />

            <Button
              variant="contained"
              onClick={updateEnergyData}
              sx={{
                backgroundColor: "#1a237e",
                "&:hover": {
                  backgroundColor: "#283593",
                },
                py: 1.5,
              }}
            >
              Update Energy Data
            </Button>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default EnergyData;
