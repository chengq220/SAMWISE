#1. endovis2017
endovis2017_category_dict = {
	"Bipolar Forceps": 1,
	"Prograsp Forceps": 2,
	"Large Needle Driver": 3,
	"Vessel Sealer": 4,
	"Grasping Retractor": 5,
    "Monopolar Curved Scissors": 6,
    "Ultrasound Probe": 7
}

endovis2017_category_rev_dict = {
	1: "Bipolar Forceps",
	2: "Prograsp Forceps",
	3: "Large Needle Driver ",
	4: "Vessel Sealer",
	5: "Grasping Retractor",
    6: "Monopolar Curved Scissors",
    7: "Ultrasound Probe"
}

endovis2017_category_verb_dict ={
    1: "Bipolar Forceps grasping and coagulating tissue",
    2: "Prograsp Forceps retracting and manipulating tissue", 
    3: "Large Needle Driver driving sutures through tissue",
    4: "Vessel Sealer sealing and dividing blood vessels",
    5: "Grasping Retractor holding back organs for exposure",
    6: "Monopolar Curved Scissors cutting and cauterizing tissue",
    7: "Ultrasound Probe scanning and visualizing internal structures"
}

endovis2017_category_rev_verb_dict = {
	1: "Bipolar Forceps grasping and coagulating tissue",
	2: "Prograsp Forceps retracting and stabilizing",
	3: "Large Needle Driver suturing and piercing",
	4: "Vessel Sealer sealing and cauterizing vessels",
	5: "Grasping Retractor holding back and exposing tissue",
    6: "Monopolar Curved Scissors cutting and dissecting",
    7: "Ultrasound Probe scanning and imaging"
}

endovis2018_category_verb_dict = {
    1: "Grasping and dissecting tissue with Bipolar Forceps",
    2: "Grasping and retracting tissue with Prograsp Forceps", 
    3: "Driving needle and suturing with Large Needle Driver",
    4: "Cutting and dissecting tissue with Monopolar Curved Scissors",
    5: "Ultrasound Probe scanning and visualizing internal structures",
    6: "Suction Instrument aspirating fluids and clearing surgical field",
    7: "Applying clips to vessels with Clip Applier"
}

endovis2017_category_descriptor_dict = {
    1: {
        "two insulated tines with small gap between tips",
        "exposed metallic distal tips (5-15mm) with insulated shafts",
        "parallel slender shafts that never cross or shear",
        "fine pointed or rounded tips for precise tissue coagulation"
    },
    2: {
        "windowed jaws with multiple interlocking teeth",
        "curved claw-like grasping tips",
        "wide jaw opening angle",
        "longer jaw length designed for tissue retraction"
    },
    3: {
        "short, robust jaws with cross-hatched tungsten carbide inserts",
        "squared-off, blunt jaw tips for needle control",
        "straight parallel shafts with precise ratchet mechanism",
        "designed for rotational wrist motions during suturing"
    },
    4: {
        "large paddle-shaped jaws",
        "thick instrument shafts",
        "jaws  angled at 15 to 45 degrees from shaft",
        "designed for sealing and cutting blood vessels"
    },
    5: {
        "broad, flat spatula-like retracting tip",
        "angled or curved blade for tissue displacement",
        "smooth, atraumatic edges to prevent tissue damage",
        "often appears as a simple metal blade extending from handle",
        "used for organ/tissue retraction rather than grasping"
    },
    6: {
        "sharp, tapered cutting blades with curved distal edges",
        "pointed tips capable of dissection and electrosurgical cutting",
        "insulated shafts except for exposed cutting surfaces",
        "classic Metzenbaum scissors shape with gentle curve"
    },
    7: {
        "Compact, sterilizable housing with smooth low-friction coating for trocar passage",
        "Miniature transducer array at distal tip (8-15 mm footprint) with forward-viewing beam geometry",
        "Integrated locking collar mechanism for secure tool mounting and depth markers along shaft",
        "Detachable autoclave-compatible cable system and color-coded orientation markers",
    }
}
