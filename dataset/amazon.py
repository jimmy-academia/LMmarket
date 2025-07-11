
from collections import defaultdict
static_feature_descriptives = defaultdict(list)
static_feature_descriptives["All_Beauty"] = {
    "quality": ["Very poor quality; product is uncomfortable, unreliable, or breaks easily", "Below average quality; some flaws, inconsistent performance or texture", "Average quality; meets basic expectations for comfort and durability", "Good quality; well-made, pleasant to use, and reliable over time", "Excellent quality; premium materials, superior feel, and long-lasting performance"],
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "quantity_size": ["Far too little or much smaller than expected", "Somewhat insufficient or smaller than described", "Adequate amount; meets basic expectations", "Generous size or portion; slightly exceeds expectations", "Perfectly sized or abundant; excellent value for the amount"],
    "absorption": ["Does not absorb at all; sits on surface and feels greasy or heavy", "Absorbs slowly or unevenly; leaves residue or tackiness", "Moderate absorption; acceptable but not ideal", "Absorbs well; leaves skin or surface feeling smooth and comfortable", "Absorbs instantly and completely; feels lightweight, clean, and efficient"],
    "fragrance": ["Unpleasant or overwhelming scent", "Noticeable but not appealing; may be too strong or artificial", "Acceptable fragrance; mild and inoffensive", "Pleasant and well-balanced scent", "Exceptional fragrance; distinctive, refined, and highly enjoyable"],
    "coverage": ["No coverage; product is sheer or ineffective at concealing", "Very light coverage; only minor evening of tone", "Medium coverage; conceals imperfections while looking natural", "High coverage; effectively hides blemishes and uneven skin tone", "Full coverage; flawless finish with strong concealing power"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Amazon_Fashion"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "fit_preference": ["Very poor fit; completely wrong size or proportions", "Somewhat uncomfortable or ill-fitting; too tight or too loose", "Acceptable fit; wearable but not ideal", "Good fit; comfortable and aligns well with body shape", "Perfect fit; feels tailored and exceptionally flattering"],
    "stretchability": ["No stretch or overly rigid; restricts movement", "Very limited stretch; feels tight or uncomfortable", "Moderate stretch; allows some flexibility but not ideal", "Good stretch; moves well with the body and feels comfortable", "Excellent stretch; highly flexible, supportive, and retains shape"],
    "skin_sensitivity": ["Caused severe irritation or allergic reaction", "Mild irritation or discomfort after use", "Generally tolerable; no major reactions", "Comfortable on skin; gentle and non-irritating", "Extremely gentle; ideal for sensitive or reactive skin"],
    "breathability_moisture_wicking": ["Extremely poor; traps heat and moisture causing discomfort", "Below average; some moisture retention and limited airflow", "Adequate; allows basic breathability and moisture control", "Good; effectively wicks moisture and feels breathable", "Excellent; keeps skin dry and cool even during intense activity"],
    "durability": ["Breaks or wears out immediately after use", "Prone to damage or deterioration with minimal use", "Average durability; lasts a reasonable amount of time", "Strong and reliable; withstands regular use well", "Exceptional durability; built to last and endure heavy use"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Appliances"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "energy_efficiency": ["Extremely power-hungry; high electricity consumption", "Below average efficiency; noticeable impact on bills", "Average efficiency; meets standard energy ratings", "Good efficiency; performs well with moderate consumption", "Exceptional efficiency; very low energy usage"],
    "durability": ["Breaks or malfunctions almost immediately", "Prone to issues after light use", "Average lifespan; lasts as expected", "Solid build; withstands regular use", "Exceptionally durable; years of trouble‑free operation"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Arts_Crafts_and_Sewing"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "material_quality": ["Very poor; tears or frays instantly", "Below average; inconsistent texture or strength", "Acceptable; usable but with minor flaws", "Good; sturdy and pleasant to work with", "Outstanding; premium feel and reliability"],
    "creativity": ["No originality; dull or clichéd", "Minimal creativity; few novel elements", "Moderately creative; some unique touches", "Highly creative; fresh and inspiring", "Exceptionally inventive; one‑of‑a‑kind designs"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Baby_Products"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "safety": ["Hazardous; poses risk of injury", "Below standard; lacks key safety features", "Meets basic safety requirements", "High safety; exceeds normal standards", "Outstanding safety; industry‑leading protection"],
    "ease_of_use": ["Extremely complicated; frustrating", "Somewhat awkward; learning curve needed", "Acceptable; functional with minor effort", "User-friendly; intuitive and smooth", "Effortless; designed for maximum convenience"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Beauty_and_Personal_Care"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "effectiveness": ["No noticeable results or adverse reactions", "Minimal improvement; inconsistent outcomes", "Moderate results; generally meets claims", "Clearly effective; visible and reliable", "Outstanding; exceeds expectations every time"],
    "fragrance": ["Unpleasant or overpowering scent", "Noticeable but unappealing or artificial", "Mild and inoffensive", "Pleasant and well-balanced", "Distinctive, refined, and highly enjoyable"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Books"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "content_quality": ["Poorly researched or incoherent", "Underdeveloped; misses depth", "Well-written; meets basic standards", "Engaging and insightful", "Exceptionally compelling and authoritative"],
    "readability": ["Very difficult to read; confusing", "Below average; awkward phrasing", "Fairly easy; some challenging sections", "Smooth and accessible", "Effortless flow; highly engaging"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["CDs_and_Vinyl"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "sound_quality": ["Distorted or muffled audio", "Below par; noticeable artifacts", "Good; clear with minor flaws", "Very good; rich and detailed", "Audiophile‑grade; pristine and full-range"],
    "packaging": ["Damaged or flimsy covers", "Basic; minimal protection or design", "Functional; adequate protection", "Well-made; visually appealing", "Collector’s quality; premium materials"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Cell_Phones_and_Accessories"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "battery_life": ["Dies almost immediately", "Short; needs frequent charging", "Average; lasts through a day", "Long; comfortable multi-day use", "Exceptional; weeks of standby time"],
    "build_quality": ["Flimsy; cracks or breaks easily", "Below average; feels cheap", "Solid; no major issues", "High quality; robust and reliable", "Flagship‑grade; premium materials & finish"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Clothing_Shoes_and_Jewelry"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "fit_preference": ["Completely ill-fitting; restrictive or baggy", "Poor; noticeable discomfort", "Acceptable; wearable with minor issues", "Good; comfortable and flattering", "Perfect; feels tailored and ideal"],
    "durability": ["Falls apart quickly", "Shows wear after few uses", "Holds up under normal wear", "Very durable; minimal wear", "Exceptionally long‑lasting"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Digital_Music"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "sound_quality": ["Heavily compressed; poor fidelity", "Below average; some artifacts", "Good; clear with minor issues", "Very good; rich and dynamic", "High‑res; studio‑quality sound"],
    "library_variety": ["Extremely limited selection", "Below average; few genres", "Decent range; familiar titles", "Wide variety; covers most tastes", "Extensive; niche to mainstream"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Electronics"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "performance": ["Very slow or unresponsive", "Below average speed", "Adequate for everyday tasks", "Fast and reliable", "Top-tier; professional‑grade performance"],
    "design": ["Bulky or unattractive", "Basic; uninspired", "Functional; acceptable aesthetics", "Stylish and modern", "Sleek, cutting‑edge design"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Gift_Cards"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "flexibility": ["Very restricted use", "Limited to few vendors", "Moderately flexible; several options", "Highly flexible; many retailers", "Universal; usable almost anywhere"],
    "ease_of_redeeming": ["Complicated process; many steps", "Somewhat confusing instructions", "Clear instructions; few hiccups", "Very straightforward", "Instant and effortless redemption"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Grocery_and_Gourmet_Food"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "freshness": ["Spoiled or stale on arrival", "Below peak freshness", "Acceptably fresh", "Very fresh; tastes recently made", "Incredibly fresh; farm‑to‑table quality"],
    "packaging": ["Damaged or leaks", "Basic; minimal protection", "Adequate; no damage", "Well‑sealed and sturdy", "Premium; preserves quality perfectly"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Handmade_Products"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "craftsmanship": ["Rough; obvious flaws", "Below average; some inconsistencies", "Competent; meets expectations", "Skilled; fine details", "Masterful; exquisite handiwork"],
    "uniqueness": ["Mass‑produced feel", "Minor unique touches", "Noticeably handmade", "Distinctive design", "One‑of‑a‑kind masterpiece"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Health_and_Household"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "safety": ["Hazardous; non‑compliant", "Below standard safety features", "Meets basic safety norms", "High safety; extra protections", "Industry‑leading safety standards"],
    "effectiveness": ["No measurable benefit", "Minimal impact", "Moderate effectiveness", "Highly effective", "Outstanding results"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Health_and_Personal_Care"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "efficacy": ["No noticeable effect", "Minor improvement", "Meets expectations", "Clear and reliable results", "Exceptional therapeutic effect"],
    "gentleness": ["Harsh; causes irritation", "Some discomfort", "Generally gentle", "Very gentle; soothing", "Ultra‑gentle; ideal for all skin types"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Home_and_Kitchen"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "functionality": ["Useless; doesn’t perform task", "Below average performance", "Performs as expected", "Highly functional; extra features", "Exceptionally versatile and effective"],
    "design": ["Ungainly or unattractive", "Basic; plain", "Acceptable aesthetics", "Stylish and thoughtful", "Beautiful, form‑and‑function masterpiece"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Industrial_and_Scientific"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "precision": ["Very inaccurate; high error rate", "Below standard precision", "Acceptable accuracy", "High precision; reliable", "Laboratory‑grade exactness"],
    "durability": ["Fails under minimal stress", "Prone to wear quickly", "Sturdy under normal use", "Very robust and long‑lasting", "Engineered for extreme conditions"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Kindle_Store"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "content_variety": ["Extremely limited selection", "Below average offerings", "Moderate library", "Wide assortment", "Comprehensive global catalog"],
    "price_value": ["Overpriced; poor value", "Slightly expensive", "Fairly priced", "Good value for money", "Excellent deals and discounts"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Magazine_Subscriptions"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "content_quality": ["Poorly written or irrelevant", "Below average articles", "Decent content mix", "High-quality, engaging pieces", "Outstanding journalism and insight"],
    "delivery_reliability": ["Constantly late or missing issues", "Occasional delays", "Generally on time", "Prompt and consistent", "Always early or exactly on schedule"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Movies_and_TV"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "video_quality": ["Blurry or pixelated", "Below HD standards", "HD quality", "4K/UHD clarity", "Cinema‑level sharpness"],
    "storyline": ["Unengaging or incoherent plot", "Weak narrative", "Solid storyline", "Compelling and well‑paced", "Masterful, unforgettable storytelling"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Musical_Instruments"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "sound_quality": ["Buzzy or muffled tones", "Below average resonance", "Good tonal clarity", "Rich and balanced sound", "Concert‑grade, professional sound"],
    "build_quality": ["Flimsy; poor craftsmanship", "Some weak points", "Adequate construction", "Solid and precise build", "Artisanal, museum‑quality workmanship"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Office_Products"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "usability": ["Very difficult to use", "Below average ergonomics", "Acceptable; minor friction", "User‑friendly and efficient", "Intuitive, seamless experience"],
    "durability": ["Breaks or jams quickly", "Prone to wear", "Sturdy under normal use", "Very reliable", "Built for heavy professional use"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Patio_Lawn_and_Garden"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "durability": ["Rusts or breaks immediately", "Wears quickly outdoors", "Holds up seasonally", "Very weather‑resistant", "Lifetime outdoor durability"],
    "aesthetics": ["Unattractive design", "Basic appearance", "Pleasant but ordinary", "Attractive and stylish", "Stunning, landscape‑enhancing look"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Pet_Supplies"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "safety": ["Hazardous to pets", "Below recommended safety", "Meets pet‑safety standards", "High safety; pet‑tested", "Veterinarian‑approved safest option"],
    "durability": ["Destroyed after one use", "Wears out quickly", "Holds up under normal play", "Very sturdy and long‑lasting", "Indestructible; built for heavy chewers"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Software"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "usability": ["Unintuitive; nearly impossible to navigate", "Clunky interface; steep learning curve", "Acceptable; some minor hiccups", "Smooth and user‑friendly", "Extremely intuitive; delight to use"],
    "reliability": ["Crashes constantly", "Frequent bugs and errors", "Generally stable", "Very reliable; rare issues", "Enterprise‑grade uptime and stability"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Sports_and_Outdoors"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "performance": ["Fails to meet basic demands", "Below expectations under stress", "Meets standard performance", "High performance; reliable in play", "Elite‑level performance"],
    "durability": ["Damages or breaks instantly", "Shows wear quickly", "Durable for casual use", "Very robust and long‑lasting", "Designed for extreme conditions"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Subscription_Boxes"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "variety": ["Same items repeatedly", "Limited selection", "Reasonable rotation", "Wide assortment of surprises", "Extremely diverse and novel items"],
    "value": ["Not worth the cost", "Below average value", "Fair value for price", "Good value; enjoyable finds", "Exceptional value; exceeds cost"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Tools_and_Home_Improvement"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "durability": ["Breaks under minimal use", "Prone to wear or failure", "Average lifespan", "Very sturdy and reliable", "Professional‑grade, lifetime durability"],
    "ease_of_use": ["Very difficult; unsafe", "Somewhat awkward handling", "Acceptable; minor learning curve", "User‑friendly; smooth operation", "Effortless; engineered for ease"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Toys_and_Games"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "entertainment_value": ["Boring; unengaging", "Limited fun factor", "Moderately entertaining", "Very fun and engaging", "Highly addictive and delightful"],
    "safety": ["Choking or injury hazard", "Below recommended safety", "Meets basic safety standards", "High safety; well‑tested", "Exceptionally safe; designed for all ages"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
static_feature_descriptives["Video_Games"] = {
    "packaging_quality": ["Damaged, flimsy, or unsafe packaging", "Low-quality or unattractive packaging; may not protect product well", "Functional and acceptable packaging; does the job", "Sturdy and well-designed packaging; looks and feels premium", "Exceptional packaging; highly durable, secure, and aesthetically impressive"],
    "quality": ["Very poor build; breaks or malfunctions quickly", "Below average quality; frequent issues or fragile parts", "Average quality; performs adequately with normal wear", "Good quality; durable and reliable over time", "Excellent quality; premium materials and long-lasting performance"],
    "shipping_speed": ["Extremely delayed, arrived much later than promised", "Slow delivery, took longer than expected", "Average speed, arrived within the estimated time", "Fast delivery, arrived earlier than expected", "Very fast, arrived much sooner than expected"],
    "graphics_quality": ["Pixelated or outdated visuals", "Below modern standards", "Decent graphics; acceptable fidelity", "High‑definition, detailed", "Cutting‑edge, photo‑realistic"],
    "gameplay": ["Unresponsive or broken mechanics", "Bugs hinder experience", "Solid gameplay; minor issues", "Very polished and fluid", "Masterful design; deeply immersive"],
    "price_sensitivity": ["Rarely considers price when choosing", "Comfortable paying for better experience", "Balances cost with quality", "Prefers cheaper budget options", "Very sensitive to price changes"],
}
