// components/LoadingScreen.tsx

import React, { useEffect, useState } from 'react';
import Image from 'next/image';
import logo from "../../public/Main Logo (light).svg";

interface LoadingScreenProps {
    onComplete: () => void;
}

const LoadingScreen: React.FC<LoadingScreenProps> = ({ onComplete }) => {
    // Local state to trigger the fade-out animation.
    const [fadeOut, setFadeOut] = useState(false);

    useEffect(() => {
        // After 4200ms, start the fade-out.
        const timer = setTimeout(() => {
            setFadeOut(true);
        }, 2600);
        return () => clearTimeout(timer);
    }, []);

    useEffect(() => {
        if (fadeOut) {
            // Once fade-out begins, wait for the animation duration (600ms) before finishing.
            const timer = setTimeout(() => {
                onComplete();
            }, 2600);
            return () => clearTimeout(timer);
        }
    }, [fadeOut, onComplete]);

    return (
        <div
            className={`fixed inset-0 flex flex-col items-center justify-center bg-gradient-to-r from-[#F6CFBE] to-[#B9DCF2] z-[9999] transition-opacity duration-500 ${fadeOut ? 'opacity-0' : 'opacity-100'
                }`}
        >
            <div className="relative flex flex-col items-center justify-center">
                {/* Animated ring drawn as an SVG */}
                <svg
                    className="absolute w-40 h-40 mr-[5px] mb-[42px]"
                    viewBox="0 0 100 100"
                    xmlns="http://www.w3.org/2000/svg"
                >
                    {/* Define the gradient */}
                    <defs>
                        <linearGradient id="flashFluentGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor="#3b82f6" />  {/* blue-500 */}
                            <stop offset="100%" stopColor="#2dd4bf" />  {/* teal-400 */}
                        </linearGradient>
                    </defs>

                    <circle
                        cx="50"
                        cy="50"
                        r="45"
                        fill="none"
                        stroke="url(#flashFluentGradient)"
                        strokeWidth="5"
                        strokeDasharray="282.743"
                        strokeDashoffset="282.743"
                        className="animate-ringFill"
                    />
                </svg>

                {/* Logo with fade-in and glow effect */}
                <Image
                    src={logo}
                    alt="main logo"
                    width={100}
                    height={100}
                    className="object-contain animate-fadeInGlow rounded-[8rem] cursor-pointer z-10"
                />

                {/* Title below the logo with a slide-up fade-in effect */}
                <h1 className="mt-4 text-3xl font-bold animate-textFadeIn text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-teal-400">
                    mnemorai
                </h1>
            </div>
        </div>
    );
};

export default LoadingScreen;