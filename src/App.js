import React, { useState } from 'react';
import InputForm from './components/InputForm';
import PredictionTable from './components/PredictionTable'; // Import PredictionTable
import './App.css';

const App = () => {
    const [predictions, setPredictions] = useState([]);

    const handleAddData = (data) => {
        const newPrediction = {
            courseCode: data.courseCode,
            students: data.students, // Include the number of students
            predictedSections: Math.ceil(data.students / 30), // Example logic for sections
        };
        setPredictions((prevPredictions) => [...prevPredictions, newPrediction]);
    };

    return (
        <div className="App">
            <div className="container">
                <h1 className="title">Course Section Forecasting</h1>
                <InputForm onAddData={handleAddData} />
                {predictions.length > 0 && <PredictionTable predictions={predictions} />}
            </div>
        </div>
    );
};

export default App;
