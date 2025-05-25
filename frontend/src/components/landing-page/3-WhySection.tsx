"use client"
import React, { useRef, useEffect, useState } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { ThreeDMarquee } from "../ui/3d-marquee";

const WhySection = () => {
  const sectionRef = useRef(null);
  const titleRef = useRef(null);
  const [shouldShowContent, setShouldShowContent] = useState(false);
  const [windowSize, setWindowSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const update = () => {
      if (typeof window !== "undefined") {
        setWindowSize({ width: window.innerWidth, height: window.innerHeight });
      }
    }
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  const isMobileDevice = React.useMemo(() => {
    return windowSize.width > 0 && windowSize.width < 768;
  }, [windowSize.width]);

  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start start", "end end"],
  });



  // Adjusted glow effect for mobile
  const glowOpacity = useTransform(
    scrollYProgress,
    isMobileDevice ? [0.25, 0.35] : [0.3, 0.4],
    [0, 0.6]
  );

  const glowScale = useTransform(
    scrollYProgress,
    isMobileDevice ? [0.25, 0.35] : [0.3, 0.4],
    [0.8, 1.1]
  );

  useEffect(() => {
    const unsubscribe = scrollYProgress.onChange(value => {
      setShouldShowContent(value >= 0.3);
    });
    return () => unsubscribe();
  }, [scrollYProgress, isMobileDevice]);

  // Sample images array
  const images = Array(16).fill("/logo.png");



  const fadeInUp = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0 }
  };



  return (
    <section
      ref={sectionRef}
      className="relative w-full h-[145vh] max-h-[1050px] lg:h-[100vh] h-[145vh] min-h-[1050px] sm:min-h-[1150px] md:min-h-[1050px] pt-12 sm:pt-16 md:pt-30"
    >

      {/* Background layer */}
      <div className="absolute inset-0 w-full h-[145vh] min-h-[1050px] sm:min-h-[1150px] md:min-h-[1050px] max-h-[1000px] sm:max-h-[1150px] lg:h-[100vh] overflow-hidden">
        {/* Background overlay - gradient */}
        <div className="absolute w-full h-full z-10 bg-gradient-to-b from-black/95 via-black/90 to-black/80" />

        {/* 3D Marquee Background */}
        <ThreeDMarquee
          className="pointer-events-none absolute inset-0 w-full h-full top-2 overflow-hidden"
          images={images}
        />

        {/* Animated glow effect */}
        <motion.div
          className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[300px] sm:w-[500px] md:w-[800px] h-[300px] sm:h-[500px] md:h-[800px] rounded-full bg-blue-500 blur-[80px] md:blur-[120px] z-5"
          style={{
            opacity: glowOpacity,
            scale: glowScale
          }}
        />
      </div>

      {/* Content */}
      <div className="relative z-20 w-full px-4 container mx-auto flex flex-col justify-start max-w-6xl pb-4">
        {/* Title with animation */}
        <motion.div
          ref={titleRef}
          initial={{ opacity: 0, y: 80 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.2 }}
          transition={{ duration: 0.8 }}
          className="flex flex-col items-center text-center  sm:mt-3"

        >
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.2 }}
            transition={{ delay: 0.2, duration: 0.8 }}
            className="mb-2 md:mb-3 text-blue-400 font-medium tracking-wider text-xs sm:text-sm md:text-base uppercase"
          >
            SUPERIOR LEARNING TECHNOLOGY
          </motion.div>

          <h2 className="text-xl sm:text-4xl md:text-6xl font-extrabold text-white mb-2">
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-teal-400">
              Why Choose
            </span>{" "}
            Our Approach
          </h2>

          <div className="h-1 w-24 sm:w-32 bg-gradient-to-r from-blue-500 to-teal-400 rounded-full my-3 sm:my-4 md:my-6"></div>

          <p className="text-gray-300 max-w-2xl text-xs sm:text-sm md:text-lg px-2">
            Our AI-driven approach to learning creates lasting connections in your brain
            that make information retrieval natural and effortless
          </p>
        </motion.div>

        {/* Comparison Content - Stack on mobile, side-by-side on desktop */}
        <motion.div className="w-full max-w-7xl mx-auto mt-8 ">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6 md:gap-10">
            {/* Traditional Learning - Top on mobile, Left on desktop */}
            <motion.div
              className="p-4 sm:p-6 md:p-8 rounded-2xl border border-gray-800 bg-gradient-to-br from-gray-900/50 to-black backdrop-blur-md group hover:border-gray-700 "
              initial={{ opacity: 0, x: -80 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, amount: 0.3 }}
              transition={{ duration: 0.8 }}
              whileHover={{ y: -5, transition: { duration: 0.3 } }}
            >
              <div className="flex flex-col justify-start items-start mb-4 sm:mb-6">
                <div className="flex items-center space-x-3 sm:space-x-4">
                  <div className="text-2xl sm:text-3xl md:text-4xl p-2 sm:p-3 bg-gray-800/50 rounded-xl">📚</div>
                  <h3 className="text-lg sm:text-xl md:text-3xl font-bold text-white">
                    Traditional Learning
                  </h3>
                </div>
                <div className="w-full h-px bg-gradient-to-r from-gray-800 via-gray-700 to-gray-800 my-3 sm:my-4 md:my-6"></div>
              </div>

              <motion.ul
                className="space-y-3 sm:space-y-4 md:space-y-5"

              >
                {[
                  "Repetitive memorization with minimal context",
                  "Generic learning materials not tailored to your preferences",
                  "Quick forgetting due to lack of meaningful associations",
                  "Boring, tedious process leading to low motivation"
                ].map((item, index) => (
                  <motion.li
                    key={index}
                    className="flex items-start gap-2 sm:gap-3 md:gap-4 text-gray-300 group-hover:text-gray-200 transition-colors duration-300"
                    variants={fadeInUp}
                  >
                    <span className="text-red-400 mt-0.5 sm:mt-1 text-base sm:text-lg font-bold">✗</span>
                    <p className="text-sm sm:text-base md:text-lg">{item}</p>
                  </motion.li>
                ))}
              </motion.ul>
            </motion.div>

            {/* Our Approach */}
            <motion.div
              className="p-4 sm:p-6 md:p-8 top-5 sm:top-0 rounded-2xl border border-blue-600/30 bg-gradient-to-br from-blue-900/20 to-blue-800/10 backdrop-blur-md relative overflow-hidden group hover:border-blue-500/50"
              initial={{ opacity: 0, x: 80 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, amount: 0.3 }}
              transition={{ duration: 0.8 }}
              whileHover={{ y: -5, transition: { duration: 0.3 } }}
            >
              {/* Background glow */}
              <div className="absolute -top-5 -right-10 w-20 h-20 sm:top-0 sm:-right-20 sm:w-40 sm:h-40 bg-blue-500/20 rounded-full blur-2xl sm:blur-3xl group-hover:bg-blue-500/30 transition-all duration-500"></div>

              <div className="flex flex-col justify-start items-start mb-4 sm:mb-6 relative z-10">
                <div className="flex items-center space-x-3 sm:space-x-4">
                  <div className="text-2xl sm:text-3xl md:text-4xl p-2 sm:p-3 bg-blue-800/30 rounded-xl bg-gradient-to-br from-blue-700/30 to-blue-900/30">🧠</div>
                  <h3 className="text-lg sm:text-xl md:text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-teal-400">
                    Our AI Approach
                  </h3>
                </div>
                <div className="w-full h-px bg-gradient-to-r from-blue-800/30 via-teal-500/30 to-blue-800/30 my-3 sm:my-4 md:my-6"></div>
              </div>

              <motion.ul
                className="space-y-3 sm:space-y-4 md:space-y-5 relative z-10"

              >
                {[
                  "Personalized mnemonics crafted for your learning style",
                  "Visual associations that connect to your existing knowledge",
                  "Spaced repetition algorithm based on your retention patterns",
                  "Engaging, game-like experience that keeps you motivated"
                ].map((item, index) => (
                  <motion.li
                    key={index}
                    className="flex items-start gap-2 sm:gap-3 md:gap-4 text-gray-200 group-hover:text-white transition-colors duration-300"
                    variants={fadeInUp}
                  >
                    <span className="text-teal-400 mt-0.5 sm:mt-1 text-base sm:text-lg font-bold">✓</span>
                    <p className="text-sm sm:text-base md:text-lg">{item}</p>
                  </motion.li>
                ))}
              </motion.ul>
            </motion.div>
          </div>

          {/* Bottom CTA */}
          <motion.div
            className="mt-6 text-center  "
            initial={{ opacity: 0, y: 80 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.5 }}
            transition={{ duration: 0.8 }}

          >
            <div className="mt-20 sm:mt-10">
              <button className="px-6 py-2.5 sm:px-8 sm:py-3 md:px-10 md:py-4  bg-gradient-to-r from-blue-600 to-teal-500 text-white font-semibold text-base sm:text-lg rounded-xl hover:shadow-lg hover:shadow-blue-500/30 hover:scale-105 transition-all duration-300 relative group overflow-hidden">
                <span className="relative z-10">Start Learning Smarter</span>
                <span className="absolute inset-0 bg-gradient-to-r from-blue-500 to-teal-400 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></span>
              </button>

              <p className="text-gray-400 mt-3 sm:mt-4 text-xs sm:text-sm">
                Join thousands of students already using our AI approach
              </p>
            </div>
          </motion.div>
        </motion.div>
      </div>

    </section>
  );
};

export default WhySection;
