import express from 'express';
import Transaction from '../models/Transactions.js';
const router = express.Router();

// Public Routes
router.get('/', async(req, res) => {
    return res.send(await Transaction.find({ userId: req.user._id }));
});

export default router;