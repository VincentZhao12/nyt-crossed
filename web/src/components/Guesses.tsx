import React, { FC } from 'react';

interface GuessesProps {
    correct: string[];
    wrong: string[];
}

const Guesses: FC<GuessesProps> = ({ correct, wrong }) => {
    return (
        <div>
            <h3>Guesses</h3>
            <table className="data-table">
                <thead>
                    <tr>
                        <th>Correct</th>
                        <th>Wrong</th>
                    </tr>
                </thead>
                <tbody>
                    {Array.from({
                        length: Math.max(correct.length, wrong.length),
                    }).map((_, index) => (
                        <tr key={index}>
                            <td>{correct[index] || ''}</td>
                            <td>{wrong[index] || ''}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default Guesses;
