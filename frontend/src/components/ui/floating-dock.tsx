/**
 * Note: Use position fixed according to your needs
 * Desktop navbar is better positioned at the bottom
 * Mobile navbar is better positioned at bottom right.
 **/

import { cn } from "@/lib/utils";
import {
    AnimatePresence,
    MotionValue,
    motion,
    useMotionValue,
    useSpring,
    useTransform,
} from "motion/react";
import { useRef, useState } from "react";

export const FloatingDock = ({
    items,
    desktopClassName,
}: {
    items: { title: string; icon: React.ReactNode }[];
    desktopClassName?: string;
    mobileClassName?: string;
}) => {
    return (
        <>
            <FloatingDockDesktop items={items} className={desktopClassName} />
        </>
    );
};



const FloatingDockDesktop = ({
    items,
    className,
}: {
    items: { title: string; icon: React.ReactNode }[];
    className?: string;
}) => {
    const mouseX = useMotionValue(Infinity);
    return (
        <motion.div
            onMouseMove={(e) => mouseX.set(e.pageX)}
            onMouseLeave={() => mouseX.set(Infinity)}
            className={cn(
                "mx-auto hidden flex h-16 gap-[10px] sm:gap-4 items-end rounded-2xl  px-4 pb-3",
                className
            )}
        >
            {items.map((item) => (
                <IconContainer mouseX={mouseX} key={item.title} {...item} />
            ))}
        </motion.div>
    );
};

function IconContainer({
    mouseX,
    title,
    icon,

}: {
    mouseX: MotionValue;
    title: string;
    icon: React.ReactNode;
}) {
    const ref = useRef<HTMLDivElement>(null);

    const distance = useTransform(mouseX, (val) => {
        const bounds = ref.current?.getBoundingClientRect() ?? { x: 0, width: 0 };

        return val - bounds.x - bounds.width / 2;
    });

    // Increase the maximum size of the container
    const widthTransform = useTransform(distance, [-150, 0, 150], [42, 84, 42]);
    const heightTransform = useTransform(distance, [-150, 0, 150], [42, 84, 42]);

    // Make the icon take up almost the entire container space
    const widthTransformIcon = useTransform(distance, [-150, 0, 150], [40, 80, 40]);
    const heightTransformIcon = useTransform(
        distance,
        [-150, 0, 150],
        [40, 80, 40]
    );

    const width = useSpring(widthTransform, {
        mass: 0.1,
        stiffness: 150,
        damping: 12,
    });
    const height = useSpring(heightTransform, {
        mass: 0.1,
        stiffness: 150,
        damping: 12,
    });

    const widthIcon = useSpring(widthTransformIcon, {
        mass: 0.1,
        stiffness: 150,
        damping: 12,
    });
    const heightIcon = useSpring(heightTransformIcon, {
        mass: 0.1,
        stiffness: 150,
        damping: 12,
    });

    const [hovered, setHovered] = useState(false);

    return (

        <motion.div
            ref={ref}
            style={{ width, height }}
            onMouseEnter={() => setHovered(true)}
            onMouseLeave={() => setHovered(false)}
            className="aspect-square rounded-full bg-gray-200 flex items-center justify-center relative"
        >
            <AnimatePresence>
                {hovered && (
                    <motion.div
                        initial={{ opacity: 0, y: 10, x: "-50%" }}
                        animate={{ opacity: 1, y: 0, x: "-50%" }}
                        exit={{ opacity: 0, y: 2, x: "-50%" }}
                        className="
                            px-2 py-1 whitespace-pre rounded-md
                            bg-white text-neutral-900 border border-gray-200
                            dark:bg-gray-800 dark:text-white dark:border-gray-700
                            absolute left-1/2 -translate-x-1/2 -top-10
                            w-fit text-base font-medium
                         "                    >
                        {title}
                    </motion.div>
                )}
            </AnimatePresence>
            <motion.div
                style={{ width: widthIcon, height: heightIcon }}
                className="flex items-center justify-center"
            >
                {icon}
            </motion.div>
        </motion.div>


    );
}
