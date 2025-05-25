"use client"
import React, { useState, useEffect, useRef } from "react";
import Button from "../ui/Button";
import { ContainerTextFlip } from "../ui/container-text-flip";
import { motion, useInView, useScroll, useTransform } from "framer-motion";
import Flashcard from "../Flashcard";
import flashe from "../../../public/Flashy_flashe.png";

const HeroSection = () => {
  // Create a ref for the section
  const sectionRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: sectionRef
  });
  const isInView = useInView(sectionRef, { once: true });
  const [isFlipped, setIsFlipped] = useState(false);
  // Add state to track if we're past the transition threshold
  const [isPastThreshold, setIsPastThreshold] = useState(false);


  const [windowSize, setWindowSize] = useState({ width: 0, height: 0 }); // ← safe on server

  useEffect(() => {
    const update = () => {
      if (typeof window !== "undefined") {
        setWindowSize({ width: window.innerWidth, height: window.innerHeight });
      }
    }
    update();                          // first run
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  // Screen size breakpoints
  const isSmallScreen = windowSize && windowSize.width < 750;  // sm
  const isMediumScreen = windowSize && windowSize.width >= 750 && windowSize.width < 1024;

  const cardData = {
    word: "Flasche",
    imageUrl: flashe.src,
    audioUrl: "",
    ipa: "flaʃə",
    verbalCue: "Imagine a bottle so flashy, it's like it's got its own built-in disco ball.",
    translation: "Bottle",
    languageCode: "de",
  };

  const transformAnimations = {
    h1Transform: useTransform(
      scrollYProgress,
      [0, 0.3],
      isSmallScreen ? ["translate(0%, -40%)", "translate(0%, -30%)"] :
        isMediumScreen ? ["translate(0%, -30%)", "translate(-10%, 70%)"] :
          ["translate(4%, -30%)", "translate(-20%, 30%)",]
    ),

    pTransform: useTransform(
      scrollYProgress,
      [0, 0.3],
      isSmallScreen ? ["translate(0%, -35%)", "translate(0%, -25%)"] :
        isMediumScreen ? ["translate(15%, -30%)", "translate(-10%, 75%)"] :
          ["translate(15%, -30%)", "translate(-20%, 40%)"]
    ),
    cardTransform: useTransform(
      scrollYProgress,
      [0, 0.3],
      isSmallScreen ? ["translate(-50%, 50%)", "translate(-50%, 100%)"] :
        isMediumScreen ? ["translate(-50%, 60%)", "translate(10%, 50%)"] :
          ["translate(-42%, 60%)", "translate(30%, 60%)"]
    ),

    titleFontSize: useTransform(
      scrollYProgress,
      [0, 0.3],
      isSmallScreen ? ["1.4rem", "1.4rem"] :
        isMediumScreen ? ["2.2rem", "1.7rem"] :
          ["2.5rem", "3rem"]
    ),
    subtitleFontSize: useTransform(
      scrollYProgress,
      [0, 0.3],
      isSmallScreen ? ["1.3rem", "0.9rem"] :
        isMediumScreen ? ["2.25rem", "1.2rem"] :
          ["2.25rem", "1.5rem"]
    ),
    contentOpacity: useTransform(scrollYProgress, [0, 0.1], [0, 1]),
    bgOpacity: useTransform(scrollYProgress, [0, 0.3], [0.3, 0.7]),
    bgWidth: useTransform(scrollYProgress, [0.8, 0.9], ["100%", "96%"]),
    bgHeight: useTransform(scrollYProgress, [0.8, 0.81, 0.9], ["100%", "70%", "60%"]),

    // Add this new animation for text color transition
    gradientTextOpacity: useTransform(
      scrollYProgress,
      [0, 0.3],
      [0, 1]
    ),
    originalTextOpacity: useTransform(
      scrollYProgress,
      [0, 0.3],
      [1, 0]
    ),
    // Add sparkles opacity animation
    sparklesOpacity: useTransform(
      scrollYProgress,
      [0, 0.15, 0.3],
      [0, 1, 0.7]
    )
  }

  useEffect(() => {
    const unsubscribe = scrollYProgress.on("change", (latest) => {
      setIsFlipped(latest > 0.1);
      // Update our threshold state based on scroll position
      setIsPastThreshold(latest > 0.1);
    });

    return () => unsubscribe();
  }, [scrollYProgress]);



  return (
    <motion.div ref={sectionRef} className="relative w-full h-[150vh]" initial={{ opacity: 0, y: -15 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: -15 }}
    >
      {/* This is the container that stays fixed in the viewport */}
      <div className="sticky top-[15%] h-screen w-full overflow-hidden flex justify-center">
        {/* Content Container */}
        <div className="absolute inset-0 w-full h-full max-h-[400px] flex items-center justify-center z-40">
          {/* Text Content - positioned absolute */}
          <motion.div
            className="absolute top-[8%] w-full max-w-xl"
          >
            {/* The title with sparkles effect */}
            <motion.div
              className={`relative mb-4 flex flex-col ${isPastThreshold && !isSmallScreen ? "text-left" : "text-center"}`}
              style={{
                transform: transformAnimations.h1Transform,
                fontSize: transformAnimations.titleFontSize,
                animationDuration: '300ms'
              }}
            >
              {/* Sparkles container */}
              <motion.div
                className="absolute w-full sm:w-[80%] lg:w-full h-[67vh] sm:h-[22rem] top-[3rem] lg:top-[4.5rem] max-h-[450px] z-0 flex justify-center md:justify-start "
                style={{ opacity: transformAnimations.sparklesOpacity, animationDuration: '300ms' }}
              >
                {/* Gradients */}
                <div className="absolute top-0 bg-gradient-to-r from-transparent via-blue-500 to-transparent h-[2px] w-3/4 blur-sm" />
                <div className="absolute top-0 bg-gradient-to-r from-transparent via-blue-500 to-transparent h-px w-3/4" />
                <div className="absolute top-0 bg-gradient-to-r from-transparent via-teal-400 to-transparent h-[3px] w-1/3 blur-sm" />
                <div className="absolute top-0 bg-gradient-to-r from-transparent via-teal-400 to-transparent h-px w-1/3" />
              </motion.div>

              {/* Actual title text */}
              <motion.h1
                className="relative font-bold text-gray-800 dark:text-white"
              >
                Learning Languages{" "}

                {/* Use the state variable for conditional rendering */}
                {!isPastThreshold ? (
                  <span>Faster</span>
                ) : (
                  <div className="md:translate-x-[-3%]">
                    <ContainerTextFlip
                      animationDuration={700}
                      interval={2000}
                      words={["Faster", "Better", "Easier", "Elevated"]}
                    />
                  </div>
                )}
              </motion.h1>


              <motion.h2
                className={`relative font-bold text-gray-500 dark:text-white-500  ${isPastThreshold ? "gradient-text" : ""}`}
                style={{ fontSize: transformAnimations.subtitleFontSize, animationDuration: '300ms' }}>
                Using AI Mnemonics
              </motion.h2>
            </motion.div>
            <motion.div className={`flex flex-col ${isPastThreshold && !isSmallScreen ? "text-left items-start" : "text-center items-center"}`}
              style={{ opacity: transformAnimations.contentOpacity, transform: transformAnimations.pTransform, animationDuration: '300ms' }}>
              <p className="text-gray-600 dark:text-white mb-8 text-xs sm:text-lg w-[70%]">
                Create personalized flashcards with <br /> powerful memory techniques that  <br /> make
                learning engaging and effective.
              </p>
              <div className="flex flex-wrap gap-4">
                <Button text="Get Started" onClick={() => { }} />
                <Button text="See Demo" onClick={() => { }} />
              </div>
            </motion.div>
          </motion.div>

          {/* Flashcard container - positioned absolute */}
          <div className="scale-[0.8] sm:scale-[1] -mt-[30rem]  ">
            <motion.div
              className="absolute left-1/2 -top-[20%] transition-all duration-300 "
              style={{ transform: transformAnimations.cardTransform }}
            >
              {/* Replaced the custom FlashCard with the imported Flashcard component */}

              <Flashcard
                key={isSmallScreen ? "sm" : isMediumScreen ? "md" : "lg"}
                card={cardData}
                isLoading={false}
                disableEdit={true}
                showFront={!isFlipped}
              />
            </motion.div>
          </div>
        </div>
      </div>
    </motion.div >
  );
};

export default HeroSection;
