import WatchListModel from '../models/WatchList.js'
import UserModel from '../models/User.js'; 


class WatchListController {
    
    static createWatchList = async (req, res) => {
      const { stockSymbol, stockName,type } = req.body;

      try {
        // Find the existing user by userId
        const existingUser = req.user;

        const newWatchListItem = new WatchListModel({
          userId: existingUser._id, // Use the _id of the existing user document
          stockSymbol: stockSymbol,
          stockName: stockName,
          Type: type?.toLowerCase() == 'c' ? "CRYPTO" : "STOCK",
        });

        await newWatchListItem.save();
        return res.status(201).send({ status: 'success', message: 'WatchList Added successfully' });
      } catch (error) {
        console.error(error);
        return  res.status(500).send({ status: 'failed', message: 'Cannot add the watchlist' });
      }
    };

    static removeWatchList = async (req,res)=>{
      const {watchlistId} = req.body;

console.log(watchlistId);
      try {

       await WatchListModel.deleteOne(watchlistId).then(()=>{

          res.send({
            status:200,
            message:"Watchlist removed successfully"
          });
        }).catch((err)=>{
          console.log(err);
        })

      }catch(err){
        console.log(err);
      }

    }

    static getWatchListData = async (req,res) =>{
        try{
          const watchlists = await WatchListModel.find({
            userId:req.user._id
          });
          return res.send(watchlists);
        }catch(err){
          console.log("Error occured while getting watchlist data")
        }
    }
  }

  export default WatchListController;