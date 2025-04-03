// SPDX-License-Identifier: MIT
pragma solidity ^0.8.27;

contract EnergyUsageTracker {
    struct User {
        uint256 energyConsumed;
        uint256 energyGenerated;
        bool hasOutstandingBill;
        uint256 outstandingAmount;
        uint256 excessEnergyCredits; // Credits earned from feeding excess energy to grid
    }

    struct Auction {
        address seller;
        uint256 amount;
        uint256 startTime;
        uint256 endTime;
        bool isActive;
        address highestBidder;
        uint256 highestBid;
        string serviceType; // Type of public service (hospital, police, etc.)
    }

    mapping(address => User) public users;
    mapping(uint256 => Auction) public auctions;
    uint256 public billingRate; // Cost per kWh
    uint256 public feedInTariffRate; // Rate for feeding energy back to grid
    uint256 public constant AUCTION_DURATION = 3600; // 1 hour
    uint256 public auctionCount;

    event EnergyDataUpdated(
        address indexed user,
        uint256 energyConsumed,
        uint256 energyGenerated
    );
    event BillCalculated(address indexed user, uint256 outstandingAmount);
    event BillSettled(address indexed user, uint256 amountPaid);
    event EnergyFedToGrid(address indexed user, uint256 amount, uint256 credit);
    event AuctionStarted(
        uint256 indexed auctionId,
        address indexed seller,
        uint256 amount,
        string serviceType
    );
    event AuctionBid(
        uint256 indexed auctionId,
        address indexed bidder,
        uint256 amount
    );
    event AuctionEnded(
        uint256 indexed auctionId,
        address indexed winner,
        uint256 amount
    );

    constructor(uint256 _billingRate, uint256 _feedInTariffRate) {
        billingRate = _billingRate;
        feedInTariffRate = _feedInTariffRate;
    }

    // Update energy data for the user
    function updateEnergyData(
        uint256 _energyConsumed,
        uint256 _energyGenerated
    ) public {
        if (
            users[msg.sender].energyConsumed == 0 &&
            users[msg.sender].energyGenerated == 0 &&
            users[msg.sender].outstandingAmount == 0
        ) {
            users[msg.sender] = User(
                _energyConsumed,
                _energyGenerated,
                false,
                0,
                0
            );
        } else {
            users[msg.sender].energyConsumed += _energyConsumed;
            users[msg.sender].energyGenerated += _energyGenerated;
        }

        emit EnergyDataUpdated(msg.sender, _energyConsumed, _energyGenerated);
    }

    // Feed excess energy back to the grid
    function feedExcessEnergyToGrid(uint256 amount) external {
        require(amount > 0, "Amount must be greater than 0");
        User storage user = users[msg.sender];
        require(
            user.energyGenerated >= user.energyConsumed + amount,
            "Insufficient excess energy"
        );

        // Calculate credit based on feed-in tariff
        uint256 credit = amount * feedInTariffRate;

        // Update user's energy status and credits
        user.energyGenerated -= amount;
        user.excessEnergyCredits += credit;

        // Emit event with credit information
        emit EnergyFedToGrid(msg.sender, amount, credit);
    }

    // Start an auction for excess energy
    function startEnergyAuction(
        uint256 amount,
        string memory serviceType
    ) external {
        require(amount > 0, "Amount must be greater than 0");
        User storage user = users[msg.sender];
        require(
            user.energyGenerated >= user.energyConsumed + amount,
            "Insufficient excess energy"
        );

        // Create new auction
        uint256 auctionId = auctionCount++;
        auctions[auctionId] = Auction({
            seller: msg.sender,
            amount: amount,
            startTime: block.timestamp,
            endTime: block.timestamp + AUCTION_DURATION,
            isActive: true,
            highestBidder: address(0),
            highestBid: 0,
            serviceType: serviceType
        });

        // Lock the energy amount
        user.energyGenerated -= amount;

        emit AuctionStarted(auctionId, msg.sender, amount, serviceType);
    }

    // Place a bid on an energy auction
    function placeBid(uint256 auctionId) external payable {
        Auction storage auction = auctions[auctionId];
        require(auction.isActive, "Auction is not active");
        require(block.timestamp <= auction.endTime, "Auction has ended");
        require(
            msg.value > auction.highestBid,
            "Bid must be higher than current highest bid"
        );

        // Refund previous highest bidder if exists
        if (auction.highestBidder != address(0)) {
            payable(auction.highestBidder).transfer(auction.highestBid);
        }

        // Update highest bid
        auction.highestBidder = msg.sender;
        auction.highestBid = msg.value;

        emit AuctionBid(auctionId, msg.sender, msg.value);
    }

    // End an auction and transfer energy to winner
    function endAuction(uint256 auctionId) external {
        Auction storage auction = auctions[auctionId];
        require(auction.isActive, "Auction is not active");
        require(block.timestamp > auction.endTime, "Auction has not ended yet");

        auction.isActive = false;

        if (auction.highestBidder != address(0)) {
            // Transfer energy to winner
            users[auction.highestBidder].energyConsumed += auction.amount;
            // Transfer payment to seller
            payable(auction.seller).transfer(auction.highestBid);
        } else {
            // If no bids, return energy to seller
            users[auction.seller].energyGenerated += auction.amount;
        }

        emit AuctionEnded(auctionId, auction.highestBidder, auction.amount);
    }

    // Calculate the outstanding bill for the user
    function calculateBill(address _user) public returns (uint256) {
        User storage user = users[_user];
        int256 netEnergy = int256(user.energyGenerated) -
            int256(user.energyConsumed);

        if (netEnergy < 0) {
            user.hasOutstandingBill = true;
            user.outstandingAmount = uint256(-netEnergy) * billingRate;
        } else {
            user.hasOutstandingBill = false;
            user.outstandingAmount = 0;
        }
        emit BillCalculated(_user, user.outstandingAmount);
        return user.outstandingAmount;
    }

    // Settle the outstanding bill
    function settleBill() public payable {
        User storage user = users[msg.sender];
        require(user.hasOutstandingBill, "No outstanding bill to pay.");
        require(
            msg.value >= user.outstandingAmount,
            "Insufficient payment to settle the bill."
        );

        uint256 amountPaid = user.outstandingAmount;
        user.hasOutstandingBill = false;
        user.outstandingAmount = 0;

        emit BillSettled(msg.sender, amountPaid);

        // Refund any excess payment
        if (msg.value > amountPaid) {
            payable(msg.sender).transfer(msg.value - amountPaid);
        }
    }

    // Get the user's current energy balance and bill status
    function getUserStatus(
        address _user
    )
        public
        view
        returns (
            uint256 consumed,
            uint256 generated,
            bool outstandingBill,
            uint256 outstandingAmount,
            uint256 excessEnergyCredits
        )
    {
        User storage user = users[_user];
        return (
            user.energyConsumed,
            user.energyGenerated,
            user.hasOutstandingBill,
            user.outstandingAmount,
            user.excessEnergyCredits
        );
    }

    // Get auction details
    function getAuctionDetails(
        uint256 auctionId
    )
        public
        view
        returns (
            address seller,
            uint256 amount,
            uint256 startTime,
            uint256 endTime,
            bool isActive,
            address highestBidder,
            uint256 highestBid,
            string memory serviceType
        )
    {
        Auction storage auction = auctions[auctionId];
        return (
            auction.seller,
            auction.amount,
            auction.startTime,
            auction.endTime,
            auction.isActive,
            auction.highestBidder,
            auction.highestBid,
            auction.serviceType
        );
    }
}
