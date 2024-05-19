import React, { FC } from 'react';
import '../styles/ConnectionsPopup.css';
import { MdClose } from 'react-icons/md';

interface PopupProps {
    isOpen: boolean;
    onClose: () => void; // New prop for closing
    onWin: (words: string[]) => void;
    onThree: (words: string[]) => void;
    onNotThree: (words: string[]) => void;
    words: string[];
}

const Popup: FC<PopupProps> = ({
    isOpen,
    onClose,
    onWin,
    onThree,
    onNotThree,
    words,
}) => {
    if (!isOpen) return null;

    return (
        <div className="popup-container">
            <div className="popup-content">
                <h3>Did this combination work or was it one away or none?</h3>
                <p>{words.length ? words.join(', ') : ''}</p>
                <div className="button-container">
                    <button onClick={() => onWin(words)}>Works</button>
                    <button onClick={() => onThree(words)}>One Away</button>
                    <button onClick={() => onNotThree(words)}>None</button>
                </div>
                <button
                    className="close-button"
                    onClick={onClose}
                    type="button"
                >
                    {<MdClose aria-label="Close" /> || 'x'}
                </button>
            </div>
        </div>
    );
};

export default Popup;
