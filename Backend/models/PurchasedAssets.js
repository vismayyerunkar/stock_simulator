import mongoose from "mongoose";
import moment from 'moment-timezone';

// Defining Schema
const assets = new mongoose.Schema({
  assetSymbol: { type: String, required: true, trim: true },
  assetName: { type: String, required: true, trim: true },
  assetPrice: { type: Number, required: true, trim: true },
  purchasedDate: {
    type: Date, 
    default: moment().tz('Asia/Kolkata').format('YYYY-MM-DD hh:mm:ss'),
  },
  assetType:{type:String,required:false},
  assetQuantity:{type:Number,required:true},
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "user", required: true }, // Use ref to reference UserModel
});

// Model
const AssetModel = mongoose.model("purchasedAssets", assets)
export default AssetModel

