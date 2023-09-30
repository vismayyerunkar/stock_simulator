import express from 'express';
const router = express.Router();
import WatchlistController from '../controllers/watchlistController.js';

// Public Routes
router.post('/createWatchList', WatchlistController.createWatchList)
router.post('/removeWatchList', WatchlistController.removeWatchList)
router.get("/watchlists",WatchlistController.getWatchListData);

export default router;
