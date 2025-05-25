import React, { ReactNode, useState } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';

interface BrowserProps {
    children: ReactNode;
    className?: string;
    showUrlBar?: boolean;
    urlText?: string;
    dark?: boolean;
}

const Browser: React.FC<BrowserProps> = ({
    children,
    className = "",
    showUrlBar = true,
    urlText = "example.com",
    dark = false
}) => {
    const [isExpanded, setIsExpanded] = useState(false);

    // Conditional classes based on dark mode
    const containerClasses = dark
        ? "border-gray-700"
        : "border-gray-300";

    const headerClasses = dark
        ? "bg-gray-800"
        : "bg-gray-200";

    const urlBarClasses = dark
        ? "bg-gray-700"
        : "bg-gray-100";

    const urlTextClasses = dark
        ? "text-gray-300"
        : "text-gray-500";

    const contentClasses = dark
        ? "bg-gray-800"
        : "bg-white";

    const handleBrowserClick = () => {
        setIsExpanded(true);
    };

    const handleOverlayClick = () => {
        setIsExpanded(false);
    };

    // Render the overlay portal only when expanded
    const overlayPortal = typeof document !== 'undefined'
        ? createPortal(
            <AnimatePresence>
                {isExpanded && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.3 }}
                        className="fixed top-0 left-0 right-0 bottom-0 w-screen h-screen bg-black bg-opacity-75 z-50 flex items-center justify-center"
                        onClick={handleOverlayClick}
                    >
                        {/* Centered browser with clean animation */}
                        <motion.div
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.9, opacity: 0 }}
                            transition={{
                                duration: 0.3,
                                ease: [0.19, 1.0, 0.22, 1.0] // Ease out expo for smooth feel
                            }}
                            className="relative w-[80%] h-[80%] max-w-6xl"
                            onClick={(e) => e.stopPropagation()}
                        >
                            <div className={`relative rounded-lg overflow-hidden border ${containerClasses} w-full h-full shadow-2xl shadow-[#97E2F9]`}>
                                {/* Browser header */}
                                <div className={`${headerClasses} h-8 flex items-center px-4 relative`}>
                                    {/* Traffic light dots with red dot as close button */}
                                    <div className="flex space-x-2">
                                        <button
                                            className="h-3 w-3 bg-red-500 rounded-full flex items-center justify-center hover:bg-red-600 transition-colors focus:outline-none"
                                            onClick={handleOverlayClick}
                                        >
                                            <span className="text-white text-[8px] font-bold">×</span>
                                        </button>
                                        <span className="h-3 w-3 bg-yellow-500 rounded-full"></span>
                                        <span className="h-3 w-3 bg-green-500 rounded-full"></span>
                                    </div>

                                    {/* URL bar */}
                                    {showUrlBar && (
                                        <div className={`absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 
                                            ${urlBarClasses} rounded-md h-5 w-1/2 flex items-center justify-center`}>
                                            <span className={`text-xs ${urlTextClasses} truncate`}>{urlText}</span>
                                        </div>
                                    )}
                                </div>

                                {/* Browser content */}
                                <div className={`flex items-start justify-start h-[calc(100%-2rem)] ${contentClasses} overflow-auto ${className}`}>
                                    {children}
                                </div>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>,
            document.body
        )
        : null;

    return (
        <>
            {/* Portal for overlay */}
            {overlayPortal}

            {/* Normal browser view */}
            <div className="relative group">
                {/* Shadow background that will animate with the group hover */}
                <div className="absolute inset-0 rounded-lg shadow-xl shadow-[#97E2F9] bg-gradient-to-r from-[#97E2F9]/20 to-blue-400/10 
                    group-hover:translate-y-[-5%] transition-all duration-300">
                </div>

                {/* Browser container that will move with the shadow on hover */}
                <div
                    className={`relative rounded-lg overflow-hidden border ${containerClasses} w-full h-full 
                        group-hover:translate-y-[-5%] transition-all duration-300 cursor-pointer`}
                    onClick={handleBrowserClick}
                >
                    {/* Browser header */}
                    <div className={`${headerClasses} h-8 flex items-center px-4 relative`}>
                        {/* Traffic light dots */}
                        <div className="flex space-x-2">
                            <span className="h-3 w-3 bg-red-500 rounded-full flex items-center justify-center cursor-pointer">
                                <span className="h-2 w-2 bg-red-600 rounded-full opacity-0 group-hover:opacity-100"></span>
                            </span>
                            <span className="h-3 w-3 bg-yellow-500 rounded-full"></span>
                            <span className="h-3 w-3 bg-green-500 rounded-full"></span>
                        </div>

                        {/* Optional: URL bar */}
                        {showUrlBar && (
                            <div className={`absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 
                                ${urlBarClasses} rounded-md h-5 w-1/2 flex items-center justify-center`}>
                                <span className={`text-xs ${urlTextClasses} truncate`}>{urlText}</span>
                            </div>
                        )}
                    </div>

                    {/* Browser content */}
                    <div className={`flex w-full h-full items-center justify-center  ${contentClasses} ${className}`}>
                        {children}
                    </div>
                </div>
            </div>
        </>
    );
};

export default Browser;