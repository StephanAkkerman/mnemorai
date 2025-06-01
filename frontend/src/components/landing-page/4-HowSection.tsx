"use client";

import React, { useRef, useEffect, useState, ReactNode } from "react";
import { motion, useScroll, useTransform, MotionValue, AnimatePresence } from "framer-motion";
import Browser from "../ui/Browser";
import Flashcard from "../Flashcard";
import flashe from "../../../public/Flashy_flashe.png";

const steps = [
  {
    threshold: 0.22,
    content: (
      <div className="w-full aspect-video overflow-hidden rounded-lg">
        <video
          src="/step1.mp4"
          preload="none"
          autoPlay
          loop
          muted
          playsInline
          className="w-full h-full object-cover"
        />
      </div>
    ),
  },
  {
    threshold: 0.5,
    content: (
      <div className="w-full aspect-video overflow-hidden rounded-lg">
        <video
          src="/step2.mp4"
          preload="none"
          autoPlay
          loop
          muted
          playsInline
          className="w-full h-full object-cover"
        />
      </div>
    ),
  },
  {
    threshold: Infinity,
    content: (
      <div className="w-full aspect-video overflow-hidden rounded-lg">
        <video
          src="/step3.mp4"
          preload="none"
          autoPlay
          loop
          muted
          playsInline
          className="w-full h-full object-cover"
        />
      </div>
    ),
  },
];

const sections = [
  {
    transformRange: [0, 0.2],
    transformValues: ["translate(0%, 0%)", "translate(0%, -300%)"],
    opacityRange: [0, 0.03, 0.17, 0.2],
    opacityValues: ["0", "1", "1", "0"],
    title: "Select Your Target Language",
    description:
      "Choose the language you want to learn from our growing collection of supported languages, with the word you want to learn",
    icon: "🌎",
    color: "from-blue-500 to-indigo-500",
  },
  {
    transformRange: [0.2, 0.5],
    transformValues: ["translateY(50px)", "translateY(0px)"],
    opacityRange: [0.2, 0.23, 0.47, 0.5],
    opacityValues: ["0", "1", "1", "0"],
    title: "Create Your Smart Flashcard",
    description:
      "Click 'Create Card' to let Mnemorai automatically translate your word, devise a personalized mnemonic, and generate a matching image.",
    icon: "🪄",
    color: "from-teal-500 to-emerald-500",
  },
  {
    transformRange: [0.5, 0.75],
    transformValues: ["translateY(50px)", "translateY(0px)"],
    opacityRange: [0.5, 0.53, 0.70, 0.75],
    opacityValues: ["0", "1", "1", "0"],
    title: "Review & Save Your Card",
    description:
      "Your mnemonic flashcard is ready! Fine-tune any details if needed, then save your masterpiece to your library.",
    icon: "💾",
    color: "from-blue-500 to-teal-400",
  },
];

type SectionBlockProps = {
  data: typeof sections[number];
  scrollYProgress: MotionValue<number>;
};


const SectionBlock: React.FC<SectionBlockProps> = ({ data, scrollYProgress }) => {
  const transformY = useTransform(
    scrollYProgress,
    data.transformRange,
    data.transformValues
  );
  const opacity = useTransform(
    scrollYProgress,
    data.opacityRange,
    data.opacityValues
  );

  // Fade-in animation for content inside the block
  const contentVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        staggerChildren: 0.1,
        duration: 0.4
      }
    }
  };

  // Determine visibility more directly for triggering inner animation
  const [isEntering, setIsEntering] = useState(false);
  useEffect(() => {
    const unsubscribe = scrollYProgress.onChange((v) => {
      setIsEntering(v >= data.opacityRange[1] && v < data.opacityRange[2]);
    });
    return () => unsubscribe();
  }, [scrollYProgress, data.opacityRange]);


  return (
    <motion.div
      style={{ opacity, y: transformY }}
      className="w-full -mt-10 md:mt-0 sm:w-4/5 md:w-3/4 lg:w-2/3 mx-auto" // Positioning handled by parent flexbox
    >
      <motion.div
        className={`bg-gradient-to-r ${data.color} rounded-2xl p-[1px] shadow-lg`}
        variants={contentVariants}
        initial="hidden"
        animate={isEntering ? "visible" : "hidden"} // Trigger based on entry
      >
        <div className="bg-gray-50 dark:bg-gray-900/90 backdrop-blur-sm rounded-2xl p-4 sm:p-6 border border-gray-200 dark:border-gray-800">
          <motion.div variants={contentVariants} className="flex items-center mb-0 sm:mb-4">
            <div className="mr-3 p-2 bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-800 dark:to-gray-900 rounded-lg">{data.icon}</div>
            <h3 className="font-bold text-sm sm:text-xl md:text-2xl bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-700 dark:from-white dark:to-gray-300">
              {data.title}
            </h3>
          </motion.div>
          <motion.div variants={contentVariants} className="h-px w-full bg-gradient-to-r from-transparent via-gray-300 dark:via-gray-700 to-transparent mb-3 sm:mb-4" />
          <motion.p variants={contentVariants} className="text-xs md:text-base text-gray-700 dark:text-gray-300">
            {data.description}
          </motion.p>
        </div>
      </motion.div>
    </motion.div>
  );
};

const HowSection: React.FC = () => {
  const sectionRef = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start start", "end end"] // Standard full-height scroll
  });

  // State for current step and content
  const [currentStep, setCurrentStep] = useState(0);
  const [browserContent, setBrowserContent] = useState<ReactNode>(steps[0].content);

  // Determine when to switch to the final card view
  const revealThreshold = 0.70; // Point at which the final card starts appearing
  const [showCard, setShowCard] = useState(false);

  useEffect(() => {
    const unsubscribe = scrollYProgress.onChange((v) => {
      // Update step content based on scroll
      const stepIndex = steps.findIndex((s) => v < s.threshold);
      const activeStep = stepIndex === -1 ? steps.length - 1 : stepIndex; // If past all thresholds, stay on last step

      if (activeStep !== currentStep) {
        setCurrentStep(activeStep);
        // Only update browser content if we are *not* showing the final card
        if (v < revealThreshold) {
          setBrowserContent(steps[activeStep].content);
        }
      }

      // Determine if we should show the final card
      setShowCard(v >= revealThreshold);
    });
    return () => unsubscribe();
    // Add currentStep to dependencies to avoid stale state in content update logic
  }, [scrollYProgress, currentStep, revealThreshold]);

  // --- Animations based on Scroll ---
  // Title animation
  const titleY = useTransform(scrollYProgress, [0, 0.0], ["0px", "50px"]);
  const titleOpacity = useTransform(scrollYProgress, [0, 0.01], [1, 1]);

  // Step Indicator Opacity (Fade out before final card)
  const stepsOpacity = useTransform(scrollYProgress, [0, 0.02, revealThreshold - 0.15, revealThreshold - 0.1], [0, 1, 1, 0]);

  // Browser animation (Scale and fade out before final card)
  const browserOpacity = useTransform(scrollYProgress, [0, 0.02, revealThreshold - 0.1, revealThreshold - 0.05], [0, 1, 1, 0]);

  // --- Flashcard Data ---
  const cardData = {
    word: "Flasche",
    imageUrl: flashe.src,
    audioUrl: "",
    ipa: "ˈflaʃə",
    verbalCue: "Imagine a Flashy bottle throwing a party with disco lights inside.",
    translation: "Bottle",
    languageCode: "de",
  };

  return (
    <section
      ref={sectionRef}
      id="how-it-works"
      className="relative w-full h-[400vh]"
    >
      {/* Sticky container takes full viewport height and centers content */}
      <div className="sticky top-0 h-screen max-h-[1000px] w-full overflow-hidden flex flex-col items-center justify-center">

        {/* Max width container for content, centered */}
        <div className="relative w-full max-w-6xl mx-auto px-4 sm:px-6 h-full flex flex-col items-center justify-center pt-16 sm:pt-20 ">

          {/* --- Title Area --- */}
          <motion.div
            className=" top-10 sm:top-30 left-0 right-0 z-20 text-center px-4"
            style={{ y: titleY, opacity: titleOpacity }}
          >
            <span className="text-xs sm:text-sm font-medium text-blue-500 dark:text-blue-400 uppercase tracking-wider">Step by step process</span>
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-gray-800 dark:text-white mt-1 sm:mt-2">
              How It <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-teal-400">Works</span>
            </h2>
            <div className="h-1 w-16 sm:w-20 bg-gradient-to-r from-blue-500 to-teal-400 mx-auto mt-2 sm:mt-4 rounded-full"></div>
          </motion.div>

          {/* --- Main Content Area (Switches between steps and final card) --- */}
          <div className="relative w-full flex-grow flex flex-col items-center justify-center mt-16 md:mt-20 min-h-[500px]">
            <AnimatePresence mode="wait">
              {!showCard ? (
                /* --- Scrolling Steps View --- */
                <motion.div
                  key="steps-view"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="w-full flex flex-col items-center  md:mt-0 space-y-12 md:space-y-8"
                >
                  {/* Step Indicators */}
                  <motion.div
                    className="flex justify-center flex-row"
                    style={{ opacity: stepsOpacity }}
                  >
                    {steps.map((_, idx) => (
                      <div key={idx} className="flex items-center">
                        <div
                          className={`w-6 h-6 sm:w-7 sm:h-7 md:w-8 md:h-8 rounded-full flex items-center justify-center text-xs sm:text-sm md:text-base transition-colors duration-300 ${idx <= currentStep
                            ? 'bg-gradient-to-r from-blue-500 to-teal-400 text-white font-semibold'
                            : 'bg-gray-200 dark:bg-gray-800 text-gray-500 dark:text-gray-400'
                            }`}
                        >
                          {idx + 1}
                        </div>
                        {idx < steps.length - 1 && (
                          <div className="w-12 md:w-20 h-0.5 mx-1 md:mx-2 relative">
                            <div className="absolute inset-0 bg-gray-200 dark:bg-gray-800 rounded-full"></div>
                            <motion.div
                              className="absolute inset-0 bg-gradient-to-r from-blue-500 to-teal-400 rounded-full"
                              initial={{ scaleX: 0 }}
                              animate={{ scaleX: idx < currentStep ? 1 : 0 }}
                              transition={{ duration: 0.3, ease: "easeInOut" }}
                              style={{ originX: 0 }}
                            />
                          </div>
                        )}
                      </div>
                    ))}
                  </motion.div>

                  {/* Browser Mockup */}
                  <motion.div
                    style={{ opacity: browserOpacity }}
                    className="w-full max-w-xs sm:max-w-sm md:max-w-md lg:max-w-lg xl:max-w-xl mx-auto"
                  >
                    <div className="relative w-full md:min-h-[250px] h-full mx-auto shadow-xl shadow-blue-500/10 rounded-lg">
                      <Browser urlText="https://mnemorai.com" dark>
                        {/* AnimatePresence for content swap */}
                        <AnimatePresence mode="wait">
                          <motion.div
                            key={currentStep}
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.2 }}
                          >
                            {browserContent}
                          </motion.div>
                        </AnimatePresence>
                      </Browser>
                    </div>
                  </motion.div>

                  {/* Explanatory Text Blocks Container */}
                  <div className="relative w-full sm:w-[80%] md:w-full h-40 sm:h-48 flex items-center justify-center">
                    {/* Position blocks absolutely within this container for smooth overlap */}
                    {sections.map((sec, idx) => (
                      <div key={idx} className="absolute inset-0 flex items-center justify-center">
                        <SectionBlock data={sec} scrollYProgress={scrollYProgress} />
                      </div>
                    ))}
                  </div>

                </motion.div>
              ) : (
                /* --- Final Card Reveal View --- */
                <motion.div
                  key="reveal-view"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.3, delay: 0.01 }} // Slight delay after steps fade
                  className="w-full  flex flex-col items-center justify-center text-center sm:mt-0 -mt-28"
                >
                  {/* Et Voilà Title */}
                  <motion.div
                    className="mb-6 md:mb-8"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                  >
                    <div className="inline-block p-2 px-4 bg-gradient-to-r from-blue-500/10 to-teal-500/10 rounded-full mb-3 sm:mb-4">
                      <h3 className="font-bold text-lg sm:text-xl md:text-2xl bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-teal-400">
                        Et voilà!
                      </h3>
                    </div>
                    <p className="text-sm sm:text-base md:text-lg text-gray-700 dark:text-gray-300 leading-relaxed">
                      Just like that you created a{" "}
                      <span className="font-semibold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-teal-400">
                        personalized
                      </span>
                      <br className="hidden sm:block" /> mnemonic-empowered flashcard!
                    </p>
                  </motion.div>


                  <div className="scale-[0.6] sm:scale-[0.7] md:scale-[1] -mt-20 sm:-mt-16 md:mt-0">
                    {/* Glow effect */}
                    <div className="absolute -inset-2 sm:-inset-3 md:-inset-4 bg-gradient-to-r from-blue-500/20 to-teal-400/20 rounded-3xl blur-lg sm:blur-xl opacity-60 animate-pulse"></div>
                    <Flashcard isLoading={false} card={cardData} className="mx-auto" />
                  </div>

                  {/* CTA Button */}
                  <motion.div
                    className="-mt-16 sm:-mt-10 md:mt-10"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.5, delay: 0.6 }}
                  >
                    <button className="px-6 md:px-8 py-2.5 md:py-3 bg-gradient-to-r from-blue-600 to-teal-500 text-white font-medium rounded-full shadow-md hover:shadow-lg hover:shadow-blue-500/30 focus:outline-none focus:ring-2 focus:ring-offset-2 
                    focus:ring-blue-500 focus:ring-offset-gray-900 transform hover:scale-105 transition-all duration-300 text-sm md:text-base"
                      onClick={() => {
                        window.location.assign("https://demo.mnemorai.com/card-gen")
                      }}>
                      Try It Now
                    </button>
                  </motion.div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowSection;
