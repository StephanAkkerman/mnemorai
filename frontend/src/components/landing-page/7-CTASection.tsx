
import React, { useRef } from "react";
import { TextGenerateEffect } from "@/components/ui/text-generate-effect";
import Button from "@/components/ui/Button";
import { motion } from "framer-motion";
import { useInView } from "framer-motion";


const CTASection = () => {
    const ref = useRef(null);
    const isInView = useInView(ref, { once: false, amount: 0.3 });

    return (
        <>
            <div ref={ref} className="relative w-full h-[4rem] flex justify-center overflow-hidden ">
                <motion.span
                    className="absolute border-t border-gradient-to-r shadow-teal-glow-xs sm:shadow-teal-glow-sm lg:shadow-teal-glow-md from-blue-500 to-teal-400 rounded-xl"
                    initial={{ width: "0%" }}
                    animate={isInView ? { width: "90%" } : { width: "0%" }}
                    transition={{ duration: 1.2, ease: "easeOut" }}
                />
            </div>

            <div className="container mx-auto px-6 text-center -mt-10 sm:mt-0 sm:mb-10" >
                <h2 className="text-3xl font-bold text-gray-800 mb-6">
                    <TextGenerateEffect words={'Start Your Language Learning Journey Today'} className="text-3xl bg-gradient-to-r from-blue-500 to-teal-400 bg-clip-text text-transparent" />
                </h2>
                <motion.div
                    initial={{ y: "50%", opacity: "0%" }}
                    animate={isInView ? { y: "0%", opacity: "100%" } : { y: "50%", opacity: "0%" }}
                    transition={{ duration: 1, ease: "easeOut" }}
                    viewport={{ once: true }}
                >
                    <Button text="Create your card now" link="https://demo.mnemorai.com/card-gen" />
                </motion.div>

            </div>
        </>)
}

export default CTASection;