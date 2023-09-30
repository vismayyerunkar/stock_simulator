import mongoose from "mongoose";
import moment from 'moment-timezone';

const goalAssestTable = new mongoose.Schema({
  goalId: { type: mongoose.Schema.Types.ObjectId, ref: "GoalTableData", required: true },
  Symbol: String,
  cagr: Number,
  marketCap: Number,
  price: Number,
  purchase: Number,
  quantity: Number,
  exp_cagr: Number,
  exp_returns: Number,
  percentage: Number,
})

export default mongoose.model('GoalAssestData', goalAssestTable);

