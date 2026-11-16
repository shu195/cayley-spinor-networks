import math, random
random.seed(195)
P = r'C:\Users\Admin\.gemini\antigravity\scratch\clean-text-dataset\dataset.txt'
out_f, out_c = [], []

def fact(t, s): out_f.append(f"FACT [{t}]: {s}")
def convo(q, a): out_c.append(f"User: {q}\nAssistant: {a}\n<|end|>")

caps = """Afghanistan:Kabul|Argentina:Buenos Aires|Austria:Vienna|Bangladesh:Dhaka|Belgium:Brussels|Bolivia:Sucre|Bulgaria:Sofia|Cambodia:Phnom Penh|Chile:Santiago|China:Beijing|Colombia:Bogota|Croatia:Zagreb|Cuba:Havana|Czech Republic:Prague|Denmark:Copenhagen|Ecuador:Quito|Ethiopia:Addis Ababa|Finland:Helsinki|France:Paris|Germany:Berlin|Ghana:Accra|Greece:Athens|Hungary:Budapest|Iceland:Reykjavik|Indonesia:Jakarta|Iran:Tehran|Iraq:Baghdad|Ireland:Dublin|Israel:Jerusalem|Italy:Rome|Jamaica:Kingston|Jordan:Amman|Kazakhstan:Astana|Laos:Vientiane|Lebanon:Beirut|Malaysia:Kuala Lumpur|Mexico:Mexico City|Mongolia:Ulaanbaatar|Morocco:Rabat|Nepal:Kathmandu|Netherlands:Amsterdam|New Zealand:Wellington|Nigeria:Abuja|Norway:Oslo|Pakistan:Islamabad|Peru:Lima|Philippines:Manila|Poland:Warsaw|Portugal:Lisbon|Romania:Bucharest|Russia:Moscow|Saudi Arabia:Riyadh|Serbia:Belgrade|Singapore:Singapore|Slovakia:Bratislava|Somalia:Mogadishu|South Africa:Pretoria|Spain:Madrid|Sri Lanka:Sri Jayawardenepura Kotte|Sudan:Khartoum|Sweden:Stockholm|Switzerland:Bern|Syria:Damascus|Tanzania:Dodoma|Thailand:Bangkok|Tunisia:Tunis|Ukraine:Kyiv|United Kingdom:London|United States:Washington, D.C.|Uruguay:Montevideo|Venezuela:Caracas|Vietnam:Hanoi|Zimbabwe:Harare|Algeria:Algiers|Angola:Luanda|Belarus:Minsk|Bhutan:Thimphu|Botswana:Gaborone|Estonia:Tallinn|Latvia:Riga|Lithuania:Vilnius|Libya:Tripoli|Madagascar:Antananarivo|Myanmar:Naypyidaw|Namibia:Windhoek|Panama:Panama City|Paraguay:Asuncion|Qatar:Doha|Senegal:Dakar|Slovenia:Ljubljana|Uganda:Kampala|Uzbekistan:Tashkent|Yemen:Sanaa|Zambia:Lusaka|Afghanistan2:Kabul""".split("|")
caps = [c.split(":") for c in caps if "2:" not in c]
for c, k in caps:
    fact("geography", f"The capital of {c} is {k}.")
    convo(f"What is the capital of {c}?", f"The capital of {c} is {k}.")

el = "H Hydrogen|He Helium|Li Lithium|Be Beryllium|B Boron|C Carbon|N Nitrogen|O Oxygen|F Fluorine|Ne Neon|Na Sodium|Mg Magnesium|Al Aluminium|Si Silicon|P Phosphorus|S Sulfur|Cl Chlorine|Ar Argon|K Potassium|Ca Calcium|Sc Scandium|Ti Titanium|V Vanadium|Cr Chromium|Mn Manganese|Fe Iron|Co Cobalt|Ni Nickel|Cu Copper|Zn Zinc|Ga Gallium|Ge Germanium|As Arsenic|Se Selenium|Br Bromine|Kr Krypton|Rb Rubidium|Sr Strontium|Y Yttrium|Zr Zirconium|Nb Niobium|Mo Molybdenum|Tc Technetium|Ru Ruthenium|Rh Rhodium|Pd Palladium|Ag Silver|Cd Cadmium|In Indium|Sn Tin|Sb Antimony|Te Tellurium|I Iodine|Xe Xenon".split("|")
for z, e in enumerate(el, 1):
    s, n = e.split(" ")
    fact("chemistry", f"{n} has atomic number {z} and chemical symbol {s}.")
    convo(f"What is the atomic number of {n.lower()}?", f"{n} has atomic number {z}, and its symbol is {s}.")
    convo(f"Which element has the symbol {s}?", f"{s} is the symbol for {n.lower()}, atomic number {z}.")

# arithmetic
for a in range(1, 21):
    for b in range(a, 21):
        fact("arithmetic", f"{a} x {b} = {a*b}.")
for a in range(1, 31):
    for b in range(a, 31, 3):
        fact("arithmetic", f"{a} + {b} = {a+b}.")
for a in range(1, 41):
    fact("arithmetic", f"{a} squared is {a*a} and {a} cubed is {a**3}.")
    fact("arithmetic", f"The square root of {a*a} is {a}.")
for n in range(0, 21):
    fact("arithmetic", f"{n} factorial is {math.factorial(n)}.")
for n in range(0, 41):
    fact("arithmetic", f"2 to the power of {n} is {2**n}.")
for n in range(0, 256, 5):
    fact("computing", f"The decimal number {n} is {bin(n)[2:]} in binary and {hex(n)[2:].upper()} in hexadecimal.")
primes = [n for n in range(2, 1000) if all(n % d for d in range(2, int(n**.5) + 1))]
fact("arithmetic", f"There are {len(primes)} prime numbers below 1000.")
for i in range(0, len(primes), 10):
    fact("arithmetic", "Primes: " + ", ".join(map(str, primes[i:i+10])) + ".")
fib = [0, 1]
while len(fib) < 40: fib.append(fib[-1] + fib[-2])
for i, f in enumerate(fib): fact("mathematics", f"Fibonacci number F({i}) is {f}.")

def roman(n):
    r = ""
    for v, s in [(1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),(100,"C"),(90,"XC"),(50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")]:
        while n >= v: r += s; n -= v
    return r
for n in range(1, 101):
    fact("language", f"The Roman numeral for {n} is {roman(n)}.")
for n in range(1, 41):
    convo(f"What is {roman(n*7 % 99 + 1)} in decimal?", f"{roman(n*7 % 99 + 1)} is {n*7 % 99 + 1}.")

# conversions
for c in range(-40, 101, 5):
    fact("units", f"{c} degrees Celsius equals {c*9/5+32:g} degrees Fahrenheit and {c+273.15:g} kelvin.")
for km in [1, 2, 5, 10, 21.0975, 42.195, 50, 100, 200, 500]:
    fact("units", f"{km:g} kilometers is about {km*0.621371:.3f} miles.")
for kg in [1, 2, 5, 10, 25, 50, 75, 100]:
    fact("units", f"{kg} kilograms is about {kg*2.20462:.2f} pounds.")
for h in range(1, 25):
    fact("units", f"{h} hours is {h*60} minutes or {h*3600} seconds.")
for d in range(1, 31):
    convo(f"How many hours are in {d} days?", f"{d} days is {d*24} hours.")
for c in range(0, 101, 4):
    convo(f"Convert {c} Celsius to Fahrenheit.", f"{c} C = {c*9/5+32:g} F, using F = C x 9/5 + 32.")
months = [("January",31),("February","28 (29 in a leap year)"),("March",31),("April",30),("May",31),("June",30),("July",31),("August",31),("September",30),("October",31),("November",30),("December",31)]
for i, (m, d) in enumerate(months, 1):
    fact("calendar", f"{m} is month number {i} of the year and has {d} days.")
    convo(f"How many days are in {m}?", f"{m} has {d} days.")
pl = [("Mercury",0.39,88),("Venus",0.72,225),("Earth",1.00,365),("Mars",1.52,687),("Jupiter",5.20,4333),("Saturn",9.58,10759),("Uranus",19.2,30687),("Neptune",30.05,60190)]
for i, (n, au, days) in enumerate(pl, 1):
    fact("astronomy", f"{n} is planet number {i} from the Sun, about {au} AU away, with an orbital period of about {days} Earth days.")
    convo(f"How long is a year on {n}?", f"A year on {n} is about {days} Earth days.")
for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 1):
    fact("language", f"The letter {ch} is number {i} of the English alphabet.")
for w, wd in zip(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], range(1, 8)):
    fact("calendar", f"{w} is day {wd} of the ISO week.")
for a in range(2, 13):
    for b in (3, 4, 6, 7, 8, 9):
        convo(f"What is {a} times {b}?", f"{a} times {b} is {a*b}.")
for n in range(20, 120, 3):
    convo(f"Is {n} prime?", f"{'Yes' if n in primes else 'No'}, {n} is {'prime' if n in primes else 'not prime'}." + ("" if n in primes else f" It is divisible by {next(d for d in range(2, n) if n % d == 0)}."))

with open(P, "r", encoding="ascii") as f: txt = f.read()
txt = txt.replace("### END OF DATASET\n", "")
with open(P, "w", encoding="ascii", newline="\n") as f:
    f.write(txt)
    f.write("\n### SECTION: FACTS (GENERATED, COMPUTED OR TABLE-DRIVEN)\n\n" + "\n".join(dict.fromkeys(out_f)) + "\n")
    f.write("\n### SECTION: CONVERSATIONS (GENERATED)\n\n" + "\n".join(out_c) + "\n\n### END OF DATASET\n")
print(len(out_f), len(out_c))
