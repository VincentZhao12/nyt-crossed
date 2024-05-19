import React, { FC } from 'react';
import '../styles/Spinner.css';

interface SpinnerProps {}

const Spinner: FC<SpinnerProps> = () => {
    return <span className="loader"></span>;
};

export default Spinner;
