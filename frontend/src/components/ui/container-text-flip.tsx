"use client";

import React, { useState, useEffect, useId } from "react";

import { motion } from "motion/react";
import { cn } from "@/lib/utils";

export interface ContainerTextFlipProps {
    /** Array of words to cycle through in the animation */
    words?: string[];
    /** Time in milliseconds between word transitions */
    interval?: number;
    /** Additional CSS classes to apply to the container */
    className?: string;
    /** Additional CSS classes to apply to the text */
    textClassName?: string;
    /** Duration of the transition animation in milliseconds */
    animationDuration?: number;
}

export function ContainerTextFlip({
    words = ["better", "modern", "beautiful", "awesome"],
    interval = 3000,
    className,
    textClassName,
    animationDuration = 700,
}: ContainerTextFlipProps) {
    const id = useId();
    const [currentWordIndex, setCurrentWordIndex] = useState(0);
    const [width, setWidth] = useState(100);
    const textRef = React.useRef(null);

    const updateWidthForWord = () => {
        if (textRef.current) {
            // @ts-expect-error needed for linting
            const textWidth = textRef.current.scrollWidth + 30;
            setWidth(textWidth);
        }
    };

    useEffect(() => {
        // Update width whenever the word changes
        updateWidthForWord();
    }, [currentWordIndex]);

    useEffect(() => {
        const intervalId = setInterval(() => {
            setCurrentWordIndex((prevIndex) => (prevIndex + 1) % words.length);
            // Width will be updated in the effect that depends on currentWordIndex
        }, interval);

        return () => clearInterval(intervalId);
    }, [words, interval]);

    return (
        <motion.h2
            layout
            layoutId={`words-here-${id}`}
            animate={{ width }}
            transition={{ duration: animationDuration / 2000 }}
            className={cn(
                "relative inline-block rounded-lg pt-4 pb-3 text-center font-bold text-black  dark:text-white",

                className,
            )}
            key={words[currentWordIndex]}
        >
            <motion.div
                transition={{
                    duration: animationDuration / 1000,
                    ease: "easeInOut",
                }}
                className={cn("inline-block", textClassName)}
                ref={textRef}
                layoutId={`word-div-${words[currentWordIndex]}-${id}`}
            >
                <motion.div className="inline-block">
                    {words[currentWordIndex].split("").map((letter, index) => (
                        <motion.span
                            key={index}
                            initial={{
                                opacity: 0,
                                filter: "blur(10px)",
                            }}
                            animate={{
                                opacity: 1,
                                filter: "blur(0px)",
                            }}
                            transition={{
                                delay: index * 0.02,
                            }}
                        >
                            {letter}
                        </motion.span>
                    ))}
                </motion.div>
            </motion.div>
        </motion.h2>
    );
}
