import AssetsModel from '../models/PurchasedAssets.js'
import Transaction from '../models/Transactions.js';
import UserModel from '../models/User.js';
import { getCurrentStockPrice } from './stockController.js';


class AssetsController {

    static purchaseAsset = async (req, res) => {

      const { assetSymbol, assetName,assetType,assetPrice,assetQuantity } = req.body;
        if(!assetName || !assetPrice || !assetSymbol || !assetType || !assetQuantity){
            res.send({
                status:400,
                message:"Please provide the required data"
            })
       }

      try {

        const existingUser = req.user;

        const existingAsset = await AssetsModel.findOne({
          assetSymbol:assetSymbol?.toLowerCase(),
        });

        console.log(existingAsset);
        if(existingAsset){

          existingAsset.assetQuantity += assetQuantity;
          await existingAsset.save();
        }else{

          const newAssetItem = new AssetsModel({
            userId: existingUser._id, // Use the _id of the existing user document
            assetSymbol: assetSymbol?.toLowerCase(),
            assetPrice:assetPrice,
            assetName: assetName?.toLowerCase(),
            assetQuantity:assetQuantity,
            assetType: assetType,
          });
          await newAssetItem.save();
        }

        // Find the existing user by userId
        console.log(existingUser._id.toString());
        // create a transaction
        const user = await UserModel.findOne({_id:req.user});
        console.log(user)
        if(user.availableTokens < assetPrice*assetQuantity){
            return res.send({
                status:200,
                message:"Cannot purchase the asset due to low funds"
            });
        }

        user.availableTokens -= assetPrice*assetQuantity;
        await user.save();

        const transaction = new Transaction({
            userId: existingUser._id,// Use the _id of the existing user document
            price:assetPrice,
            type:"BUY",
            assetType:assetType,
            symbol:assetSymbol,
            quantity:assetQuantity,
        });
        await transaction.save();
        return res.status(201).send({ status: 'success', message: 'Asset Purchased successfully' });
      } catch (error) {
        console.error(error);
        return  res.status(500).send({ status: 'failed', message: 'Cannot purchase asset' });
      }
    };

    static sellAsset = async (req,res)=>{
        let {assetId,assetQuantity,currentPrice} = req.body;
        assetQuantity = parseFloat(assetQuantity);
        
        try{

            const asset = await AssetsModel.findById({_id:assetId});
            console.log(asset?.assetSymbol);
            if(!asset){
                return res.send({
                    status:400,
                    message:"Asset not found"
                })
            }

            const user = await UserModel.findById({_id:req.user._id});
            if(!user){
                return res.send({
                    status:500,
                    message:"Failed to perform the operation"
                })
            }

            // const currentPrice= await getCurrentStockPrice(asset?.assetSymbol);
            console.log("curr spri : ",asset,currentPrice)
            user.availableTokens += assetQuantity * currentPrice
            await user.save();


            if(asset && asset.assetQuantity - assetQuantity > 0){
              asset.assetQuantity -= assetQuantity;
             await asset.save();
            }else{
              await AssetsModel.deleteOne({_id:assetId});
            }
           
            const transaction = new Transaction({
              userId: user._id,// Use the _id of the existing user document
              price:currentPrice,
              type:"SELL",
              symbol:asset.assetSymbol,
              assetType:asset.assetType,
              quantity:asset.assetQuantity,
            });

            await transaction.save();
            // create a transaction
            return res.send({
                status:200,
                message:"Asset sold successfully"
            });
          }catch(err){
            console.log("Error occured while getting assets data",err)
          }   
    }

    static getPurchasedAssets = async (req,res) =>{
        console.log(req.user);
        try{
          const assets = await AssetsModel.find({
            userId:req.user._id
          });

          return res.send(assets);
        }catch(err){
          console.log("Error occured while getting assets data")
        }
    }
  }
export default AssetsController;

