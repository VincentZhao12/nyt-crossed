import React, { FC } from 'react';
import { FaExclamationTriangle } from 'react-icons/fa';
import '../styles/Error.css';

interface ErrorComponentProps {
    errorMessage: string;
}

const ErrorComponent: FC<ErrorComponentProps> = ({ errorMessage }) => {
    return (
        <div className="error">
            <FaExclamationTriangle className="error-icon" />
            <p className="error-message">{errorMessage}</p>
        </div>
    );
};

export default ErrorComponent;
