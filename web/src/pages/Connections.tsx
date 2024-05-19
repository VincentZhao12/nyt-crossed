import React, { ChangeEvent, ClipboardEvent, FC, useState } from 'react';
import '../styles/Connections.css';
import { MdFileUpload } from 'react-icons/md';
import { createWorker } from 'tesseract.js';
import Spinner from '../components/Spinner';
import ConnectionsOptions from '../components/ConnectionsOptions';
import ConnectionsPopup from '../components/ConnectionsPopup';
import Guesses from '../components/Guesses';
import ErrorComponent from '../components/ErrorComponent';

interface ConnectionsProps {}

export interface ConnectionAnswer {
    words: string[];
    similarity: number;
}

const Connections: FC<ConnectionsProps> = () => {
    const [text, setText] = useState<string>();
    const [suggestions, setSuggestions] = useState<ConnectionAnswer[]>();
    const [loading, setLoading] = useState<boolean>(false);
    const [words, setWords] = useState<string[]>([]);
    const [correct, setCorrect] = useState<string[]>([]);
    const [wrong, setWrong] = useState<string[]>([]);
    const [error, setError] = useState<string>();
    const [options, setOptions] = useState<string[]>([]);

    const onPaste = (e: ClipboardEvent<HTMLTextAreaElement>) => {
        const file = e.clipboardData.files[0];

        if (file) {
            e.preventDefault();
            ocr(file);
        }
    };

    const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;

        if (!files) return;

        ocr(files[0]);
    };

    const ocr = async (file: File) => {
        const url = URL.createObjectURL(file);

        const worker = await createWorker('eng');
        const ret = await worker.recognize(url);

        let text = ret.data.text;
        text = text.toUpperCase().split(/\s+/).join(', ');
        setText(text);

        await worker.terminate();
    };

    const onTextChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
        setText(e.target.value);
    };

    const findConnections = async () => {
        if (!text) return;

        setError('');
        setCorrect([]);
        setWrong([]);
        setSuggestions([]);

        setLoading(true);

        const myHeaders = new Headers();
        myHeaders.append('Content-Type', 'application/json');

        const words = text
            .toUpperCase()
            .split(',')
            .map((word) => word.trim())
            .filter((word) => word.length);

        setOptions(words);

        const raw = JSON.stringify({ words });

        const requestOptions = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: raw,
        };

        try {
            const response = await fetch(
                'http://127.0.0.1:5000/connections',
                requestOptions,
            );
            if (!response.ok) {
                throw await response.text();
            }
            const suggestions = await response.json();

            setSuggestions(suggestions);
        } catch (error) {
            if (typeof error === 'string') setError(error);
            else setError('Unknown error has occurred');
        }
        setLoading(false);
    };

    const win = () => {
        const newSuggestions = suggestions?.filter((answer) => {
            for (let i = 0; i < answer.words.length; i++) {
                if (words.includes(answer.words[i])) return false;
            }

            return true;
        });

        const newOptions = options.filter((word) => !words.includes(word));
        setOptions(newOptions);

        setCorrect([...correct, words.join(', ')]);

        setSuggestions(newSuggestions);
        setWords([]);
    };

    const oneAway = () => {
        const newSuggestions = suggestions
            ?.map((suggestion) => {
                let matchCount = suggestion.words.reduce(
                    (accumulator, word) =>
                        words.includes(word) ? accumulator + 1 : accumulator,
                    0,
                );

                if (matchCount === 4) {
                    return { ...suggestion, similarity: -100 };
                } else if (matchCount === 3) {
                    return {
                        ...suggestion,
                        similarity: 1.2 * suggestion.similarity,
                    };
                }
                return suggestion;
            })
            .filter((suggestion) => suggestion.similarity !== -100)
            .sort((a, b) => b.similarity - a.similarity);

        setWrong([...wrong, words.join(', ')]);
        setSuggestions(newSuggestions);
        setWords([]);
    };

    const notOneAway = () => {
        const newSuggestions = suggestions?.filter((suggestion) => {
            let matchCount = suggestion.words.reduce(
                (accumulator, word) =>
                    words.includes(word) ? accumulator + 1 : accumulator,
                0,
            );

            if (matchCount >= 3) {
                return false;
            }
            return true;
        });

        setWrong([...wrong, words.join(', ')]);
        setSuggestions(newSuggestions);
        setWords([]);
    };

    return (
        <div>
            <h2 className="heading">Connections Assistant</h2>
            <div className="container">
                <div className="left">
                    <div className="pane">
                        {error && <ErrorComponent errorMessage={error} />}
                        <div className="label">
                            <input
                                id="fileInput"
                                type="file"
                                title="connections-grid"
                                onChange={onFileChange}
                                accept="image/*"
                            />
                            <label
                                htmlFor="fileInput"
                                className="upload-button"
                            >
                                <div className="hstack">
                                    <MdFileUpload size={25} />
                                    Upload Image of Connetions Grid
                                </div>
                            </label>
                        </div>
                        <div>
                            Or type words comma seperated below or paste file
                            here
                        </div>
                        <textarea
                            title="Connections Words"
                            value={text}
                            onChange={onTextChange}
                            onPasteCapture={onPaste}
                        />
                        <button onClick={findConnections}>
                            Get Suggestions
                        </button>
                    </div>
                    {correct.length + wrong.length ? (
                        <div className="pane">
                            <Guesses correct={correct} wrong={wrong} />
                        </div>
                    ) : (
                        ''
                    )}
                </div>
                {(loading || (suggestions?.length || 0) > 0) && (
                    <div className="right">
                        {loading ? (
                            <>
                                <Spinner />
                                <p>
                                    Loading Connections. This can take 10-15
                                    seconds
                                </p>
                            </>
                        ) : (
                            suggestions?.length && (
                                <>
                                    <div className="pane">
                                        <ConnectionsOptions
                                            data={suggestions}
                                            onSelect={(words) =>
                                                setWords(words)
                                            }
                                            words={options}
                                        />
                                    </div>
                                    <ConnectionsPopup
                                        isOpen={words.length !== 0}
                                        onWin={win}
                                        onThree={oneAway}
                                        onNotThree={notOneAway}
                                        onClose={() => setWords([])}
                                        words={words}
                                    />
                                </>
                            )
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default Connections;
