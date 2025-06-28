import React from 'react';
import { NavLink } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav className="navbar">
      <NavLink to="/" className="nav-logo">
        ⚽ Football Analyst AI
      </NavLink>
      <ul className="nav-menu">
        <li className="nav-item">
          <NavLink to="/" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            Home
          </NavLink>
        </li>
        {/* <li className="nav-item">
          <NavLink to="/project" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            Project
          </NavLink>
        </li> */}
        <li className="nav-item">
          <NavLink to="/solution" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            Solution
          </NavLink>
        </li>
      </ul>
    </nav>
  );
};

export default Navbar;