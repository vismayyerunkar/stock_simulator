import express from 'express';
const router = express.Router();
import * as stockController from "../controllers/stockController.js";

// Protected Routes
router.get('/getCurrentStockPrice',stockController.getCurrentStockPrice)
router.get('/getStockPriceInRange', stockController.getStockPriceBetweenDateRange)
router.get('/getTopGainersLoosers',stockController.getGainersAndLoosers)


export default router