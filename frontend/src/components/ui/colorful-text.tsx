"use client";
import React, { useEffect, useState, useRef } from "react";
import { motion } from "framer-motion";

export function ColourfulText({ text }: { text: string }) {
    // Colors representing the blue-500 to teal-400 gradient
    const colors = [
        "rgb(59, 130, 246)",  // blue-500
        "rgb(56, 142, 239)",
        "rgb(54, 154, 232)",
        "rgb(52, 166, 225)",
        "rgb(50, 178, 218)",
        "rgb(48, 190, 211)",
        "rgb(46, 202, 204)",
        "rgb(45, 208, 197)",
        "rgb(45, 212, 191)",  // teal-400
        "rgb(45, 212, 191)",  // repeated to maintain 10 colors
    ];

    const [currentColors, setCurrentColors] = useState(colors);
    const [count, setCount] = useState(0);
    const [hasAnimated, setHasAnimated] = useState(false);
    const animationCompleted = useRef(false);

    useEffect(() => {
        // Only run the animation once when the component mounts
        if (!hasAnimated) {
            // Set a timeout to match the animation duration (0.5s) plus the staggered delay
            // for all characters (index * 0.05s for each character)
            const maxDelay = text.length * 0.05;
            const animationDuration = 0.5;
            const totalDuration = (maxDelay + animationDuration) * 1000;

            // Schedule the random color change once
            const timeout = setTimeout(() => {
                const shuffled = [...colors].sort(() => Math.random() - 0.5);
                setCurrentColors(shuffled);
                setCount((prev) => prev + 1);
                setHasAnimated(true);
                animationCompleted.current = true;
            }, totalDuration);

            return () => clearTimeout(timeout);
        }
    }, [colors, text.length, hasAnimated]);

    return text.split("").map((char, index) => (
        <motion.span
            key={`${char}-${count}-${index}`}
            initial={{
                y: 0,
            }}
            animate={{
                color: currentColors[index % currentColors.length],
                y: hasAnimated ? 0 : [0, -3, 0],
                scale: hasAnimated ? 1 : [1, 1.01, 1],
                filter: hasAnimated ? "blur(0px)" : ["blur(0px)", `blur(5px)`, "blur(0px)"],
                opacity: hasAnimated ? 1 : [1, 0.8, 1],
            }}
            transition={{
                duration: hasAnimated ? 0 : 0.5,
                delay: hasAnimated ? 0 : index * 0.05,
            }}
            className="inline-block whitespace-pre font-sans tracking-tight"
        >
            {char}
        </motion.span>
    ));
}