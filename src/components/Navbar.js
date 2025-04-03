import React from "react";
import { AppBar, Toolbar, Typography, Button, Box } from "@mui/material";
import { useNavigate } from "react-router-dom";
import AccountBalanceWalletIcon from "@mui/icons-material/AccountBalanceWallet";
import DashboardIcon from "@mui/icons-material/Dashboard";
import BoltIcon from "@mui/icons-material/Bolt";
import AccountBalanceIcon from "@mui/icons-material/AccountBalance";

const Navbar = ({ walletAddress, onConnectWallet }) => {
  const navigate = useNavigate();

  return (
    <AppBar
      position="sticky"
      sx={{
        backgroundColor: "#1a1a1a",
        borderBottom: "1px solid #333333",
        boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
      }}
    >
      <Toolbar>
        <Typography
          variant="h6"
          component="div"
          sx={{
            flexGrow: 1,
            cursor: "pointer",
            fontWeight: 600,
            background: "linear-gradient(45deg, #4f83cc, #60a5fa)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            fontSize: "1.5rem",
          }}
          onClick={() => navigate("/")}
        >
          Energy Usage Tracker
        </Typography>
        <Box sx={{ display: "flex", gap: 1 }}>
          <Button
            color="inherit"
            onClick={() => navigate("/")}
            startIcon={<DashboardIcon />}
            sx={{
              backgroundColor: "rgba(79, 131, 204, 0.1)",
              "&:hover": { backgroundColor: "rgba(79, 131, 204, 0.2)" },
              borderRadius: "8px",
              textTransform: "none",
            }}
          >
            Dashboard
          </Button>
          <Button
            color="inherit"
            onClick={() => navigate("/energy-data")}
            startIcon={<BoltIcon />}
            sx={{
              backgroundColor: "rgba(79, 131, 204, 0.1)",
              "&:hover": { backgroundColor: "rgba(79, 131, 204, 0.2)" },
              borderRadius: "8px",
              textTransform: "none",
            }}
          >
            Energy Data
          </Button>
          <Button
            color="inherit"
            onClick={() => navigate("/billing")}
            startIcon={<AccountBalanceIcon />}
            sx={{
              backgroundColor: "rgba(79, 131, 204, 0.1)",
              "&:hover": { backgroundColor: "rgba(79, 131, 204, 0.2)" },
              borderRadius: "8px",
              textTransform: "none",
            }}
          >
            Billing
          </Button>
          {walletAddress ? (
            <Button
              color="inherit"
              startIcon={<AccountBalanceWalletIcon />}
              sx={{
                backgroundColor: "rgba(67, 160, 71, 0.1)",
                "&:hover": { backgroundColor: "rgba(67, 160, 71, 0.2)" },
                borderRadius: "8px",
                textTransform: "none",
                ml: 2,
              }}
            >
              {`${walletAddress.slice(0, 6)}...${walletAddress.slice(-4)}`}
            </Button>
          ) : (
            <Button
              color="inherit"
              onClick={onConnectWallet}
              startIcon={<AccountBalanceWalletIcon />}
              sx={{
                backgroundColor: "rgba(67, 160, 71, 0.1)",
                "&:hover": { backgroundColor: "rgba(67, 160, 71, 0.2)" },
                borderRadius: "8px",
                textTransform: "none",
                ml: 2,
              }}
            >
              Connect Wallet
            </Button>
          )}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;
