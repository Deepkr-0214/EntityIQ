"""Quick test of transliteration fix for cross-script matching."""
import sys
from pathlib import Path
PROJECT_ROOT = Path('d:/Amazon/business_entity_resolution')
sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing.normalization import normalize_record

pairs = [
    ('Royal Consulting Private Limited', '\u0930\u0949\u092f\u0932 \u0915\u0902\u0938\u0932\u094d\u091f\u093f\u0902\u0917 \u092a\u094d\u0930\u093e\u0907\u0935\u0947\u091f \u0932\u093f\u092e\u093f\u091f\u0947\u0921'),
    ('Red Agro Private Limited', '\u09b0\u09c7\u09a1 \u0985\u09cd\u09af\u09be\u0997\u09cd\u09b0\u09cb \u09aa\u09cd\u09b0\u09be\u0987\u09ad\u09c7\u099f \u09b2\u09bf\u09ae\u09bf\u099f\u09c7\u09a1'),
    ('Eastern Consultants', '\u0908\u0938\u094d\u091f\u0930\u094d\u0928 \u0915\u0902\u0938\u0932\u094d\u091f\u0947\u0902\u091f\u094d\u0938'),
    ('Shakti Trading Limited', '\u0936\u0915\u094d\u0924\u093f \u091f\u094d\u0930\u0947\u0921\u093f\u0902\u0917 \u0932\u093f\u092e\u093f\u091f\u0947\u0921'),
]

for s1_name, mx_name in pairs:
    s1 = normalize_record('S1-1', s1_name, 'test addr', 'india')
    mx = normalize_record('S2-1', mx_name, 'test addr', 'india')
    tok_overlap = s1['ntoks'] & mx['ntoks']
    ft_match = s1['ft'] == mx['ft']
    pfx_match = s1['npfx'] == mx['npfx']
    print("S1: " + s1_name[:50])
    print("  S1 na=" + s1['na'] + "  Mx na=" + mx['na'])
    print("  ft_match=" + str(ft_match) + "  pfx_match=" + str(pfx_match))
    print("  S1 ft=" + s1['ft'] + "  Mx ft=" + mx['ft'])
    print("  Token overlap: " + str(tok_overlap))
    print()
