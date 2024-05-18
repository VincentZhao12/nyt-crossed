import React, { FC } from 'react';
import '../styles/Home.css';
import { useNavigate } from 'react-router-dom';

interface HomeProps {}

const Home: FC<HomeProps> = () => {
    const navigate = useNavigate();

    return (
        <div className="home">
            <h1>Welcome to NYT Crossed</h1>
            <p>
                Your personal assistant in helping you solve NYT Connections and
                Wordles
            </p>
            <div className="actions">
                <button
                    onClick={() => navigate('/connections')}
                    className="connections"
                >
                    Play Connections
                </button>
                <button onClick={() => navigate('/wordle')} className="wordle">
                    Play Wordle
                </button>
            </div>
        </div>
    );
};

export default Home;
