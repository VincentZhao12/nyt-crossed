import React, { FC, useState } from 'react';
import { ConnectionAnswer } from '../pages/Connections';
import '../styles/ConnectionsOptions.css';

interface ConnectionsOptionsProps {
    data?: ConnectionAnswer[];
    onSelect: (words: string[]) => void;
    words?: string[];
}

const ConnectionsOptions: FC<ConnectionsOptionsProps> = ({
    data,
    onSelect,
    words,
}) => {
    const [word1, setWord1] = useState<string>('');
    const [word2, setWord2] = useState<string>('');
    const [word3, setWord3] = useState<string>('');
    const [word4, setWord4] = useState<string>('');

    const checkWords = () => {
        if (!words) return;
        if (words.length || 0 < 4) {
            return;
        }

        const count = [word1, word2, word3, word4].reduce((accum, word) => {
            if (words.includes(word.toUpperCase())) return accum + 1;
            return accum;
        }, 0);

        if (count === 4)
            onSelect(
                [word1, word2, word3, word4].map((word) => word.toUpperCase()),
            );
    };

    return (
        <div>
            <h3>Connections Suggestions</h3>
            <p>Select the row that you make your guess with</p>
            <table className="data-table">
                <thead>
                    <tr className="no-click">
                        <th>Word 1</th>
                        <th>Word 2</th>
                        <th>Word 3</th>
                        <th>Word 4</th>
                        <th>Score</th>
                    </tr>
                </thead>
                <tbody>
                    <tr className="row" onClick={checkWords}>
                        <td>
                            <input
                                type="text"
                                title="word1"
                                value={word1}
                                onChange={(e) => setWord1(e.target.value)}
                            />
                        </td>
                        <td>
                            <input
                                type="text"
                                title="word2"
                                value={word2}
                                onChange={(e) => setWord2(e.target.value)}
                            />
                        </td>
                        <td>
                            <input
                                type="text"
                                title="word3"
                                value={word3}
                                onChange={(e) => setWord3(e.target.value)}
                            />
                        </td>
                        <td>
                            <input
                                type="text"
                                title="word4"
                                value={word4}
                                onChange={(e) => setWord4(e.target.value)}
                            />
                        </td>
                        <td></td>
                    </tr>
                    {data &&
                        data.slice(0, 20).map((rowData, i) => (
                            <tr
                                className="row"
                                key={i}
                                onClick={() => onSelect(rowData.words)}
                            >
                                <td>{rowData.words[0]}</td>
                                <td>{rowData.words[1]}</td>
                                <td>{rowData.words[2]}</td>
                                <td>{rowData.words[3]}</td>
                                <td>
                                    {Math.floor(rowData.similarity * 1000) /
                                        100}
                                </td>
                            </tr>
                        ))}
                </tbody>
            </table>
        </div>
    );
};

export default ConnectionsOptions;
