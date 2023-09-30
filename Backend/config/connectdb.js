import mongoose from 'mongoose';

const connectDB = async (DATABASE_URL) => {
  try {
    const DB_OPTIONS = {
      dbName: "ritchin"
    }
    await mongoose.connect(DATABASE_URL, DB_OPTIONS)
    console.log('Connected to database ')
  } catch (error) {
    console.log("Error connecting to the database");
    console.log(error)
  }
}

export default connectDB;