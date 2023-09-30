import mongoose from "mongoose";

const GoalTable =  new mongoose.Schema({
    userId: { type: mongoose.Schema.Types.ObjectId, ref: "user", required: true },
    assests:[{
        type: mongoose.Schema.Types.ObjectId, 
        ref: "GoalAssestData",
        required:false
    }],
    crypto_expected_return: Number,
    crypto_percentage: Number,
    future_value: Number,
    stocks_expected_return: Number,
    stocks_percentage: Number,
    total_expected_return: Number,
    total_purchase: Number,
    years: Number,
});

export default mongoose.model('GoalTableData', GoalTable);






