// src/components/TrainModelButton.js
import React from 'react';
import * as tf from '@tensorflow/tfjs';

const TrainModelButton = ({ data, onModelTrained }) => {
    const trainModel = async () => {
        const xs = tf.tensor(data.map(d => d.year)); // Assuming you have a 'year' field
        const ys = tf.tensor(data.map(d => d.students)); // Assuming you have a 'students' field

        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

        model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

        await model.fit(xs, ys, { epochs: 100 });
        onModelTrained(model);
    };

    return <button onClick={trainModel}>Train Model</button>;
};

export default TrainModelButton;