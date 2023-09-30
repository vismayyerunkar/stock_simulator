import mongoose from "mongoose";

// Defining Schema
const watchlistSchema = new mongoose.Schema({
    userId: { type: mongoose.Schema.Types.ObjectId, ref: "user", required: true }, // Use ref to reference UserModel
    stockSymbol: { type: String, required: true},
    stockName: { type: String, required: false},
    Type: { type: String, required: true },
});

// Model
export default mongoose.model("watchlist", watchlistSchema);
