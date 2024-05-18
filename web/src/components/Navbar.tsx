import React, { FC } from 'react';
import { Link } from 'react-router-dom';
import '../styles/Navbar.css';

interface NavbarProps {}

const Navbar: FC<NavbarProps> = () => {
    return (
        <div className="nav">
            <h1>NYT Crossed</h1>
            <Link to="/connections">
                <h3 id="connections-link">Connections</h3>
            </Link>
            <Link to="/wordle">
                <h3 id="wordle-link">Wordle</h3>
            </Link>
        </div>
    );
};

export default Navbar;
