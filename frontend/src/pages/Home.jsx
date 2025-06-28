// src/pages/Home.jsx
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './Home.css';

const Home = () => {
  const navigate = useNavigate();

  const [currentSlide, setCurrentSlide] = useState(0);

  const slides = [
    {
      id: 1,
      image: "https://images.unsplash.com/photo-1574629810360-7efbbe195018?q=80&w=1993&auto=format&fit=crop&ixlib=rb-4.0.3",
      title: "AI-Powered Player Detection",
      subtitle: "Advanced Computer Vision for Real-time Player Tracking",
      description: "Our AI system automatically identifies and tracks every player on the field with precision accuracy"
    },
    {
      id: 2,
      image: "https://images.unsplash.com/photo-1551698618-1dfe5d97d256?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3",
      title: "Tactical Analysis & Insights",
      subtitle: "Deep Football Analytics with Machine Learning",
      description: "Generate comprehensive tactical insights, team formations, and strategic patterns from video analysis"
    },
    {
      id: 3,
      image: "https://images.unsplash.com/photo-1508098682722-e99c43a406b2?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3",
      title: "Real-time Match Intelligence",
      subtitle: "Live Performance Metrics & Heat Maps",
      description: "Transform raw match footage into actionable intelligence with our cutting-edge AI technology"
    }
  ];

  // Auto-slide functionality
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % slides.length);
    }, 5000); // Change slide every 5 seconds

    return () => clearInterval(timer);
  }, [slides.length]);

  const goToSlide = (index) => {
    setCurrentSlide(index);
  };

  const nextSlide = () => {
    setCurrentSlide((prev) => (prev + 1) % slides.length);
  };

  const prevSlide = () => {
    setCurrentSlide((prev) => (prev - 1 + slides.length) % slides.length);
  };

  return (
    <div className="home-container">
      <div className="hero-carousel">
        {slides.map((slide, index) => (
          <div
            key={slide.id}
            className={`slide ${index === currentSlide ? 'active' : ''}`}
            style={{
              backgroundImage: `linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.6)), url(${slide.image})`
            }}
          >
            <div className="slide-content">
              <div className="slide-text">
                <h1 className="slide-title">{slide.title}</h1>
                <h2 className="slide-subtitle">{slide.subtitle}</h2>
                <p className="slide-description">{slide.description}</p>
                <div className="slide-actions">
                  <button className="cta-button primary" onClick={() => navigate('/solution')}>
                    🚀 Try Our AI Analysis
                  </button>
                </div>
              </div>
            </div>
          </div>
        ))}

        {/* Navigation Controls */}
        <button className="nav-btn prev" onClick={prevSlide}>
          ‹
        </button>
        <button className="nav-btn next" onClick={nextSlide}>
          ›
        </button>

        {/* Slide Indicators */}
        <div className="slide-indicators">
          {slides.map((_, index) => (
            <button
              key={index}
              className={`indicator ${index === currentSlide ? 'active' : ''}`}
              onClick={() => goToSlide(index)}
            />
          ))}
        </div>

        {/* Progress Bar */}
        {/* <div className="progress-bar">
          <div
            className="progress-fill"
            style={{
              width: `${((currentSlide + 1) / slides.length) * 100}%`
            }}
          />
        </div> */}
      </div>

      {/* Features Section */}
      <section className="features-section">
        <div className="container">
          <h2 className="features-title">Powered by Advanced AI Technology</h2>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">🎯</div>
              <h3>Player Detection</h3>
              <p>Accurately identify and track all players in real-time using deep learning models</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">⚽</div>
              <h3>Ball Tracking</h3>
              <p>Precise ball detection and trajectory analysis throughout the entire match</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">📊</div>
              <h3>Tactical Analysis</h3>
              <p>Generate comprehensive tactical insights and team formation analysis</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🎮</div>
              <h3>Interactive Radar</h3>
              <p>Dynamic tactical radar view with real-time positioning and movement patterns</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;