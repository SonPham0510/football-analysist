import React from 'react';
import './ErrorMessage.css';

const ErrorMessage = ({ error, onDismiss }) => {
    if (!error) return null;

    return (
        <div className="error-container">
            <div className="error-content">
                <div className="error-icon">⚠️</div>
                <div className="error-text">
                    <h4>Processing Error</h4>
                    <p>{error}</p>
                </div>
                {onDismiss && (
                    <button onClick={onDismiss} className="error-dismiss">
                        ✕
                    </button>
                )}
            </div>
        </div>
    );
};

export default ErrorMessage;
