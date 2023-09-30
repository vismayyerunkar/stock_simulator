import express from 'express';
const router = express.Router();
import AssetsController from '../controllers/assetsController.js';

// Public Routes
router.post('/purchaseAsset', AssetsController.purchaseAsset)
router.post('/sellAsset', AssetsController.sellAsset)
router.get("/",AssetsController.getPurchasedAssets);

export default router;
