import os
import csv
import re
import argparse
import glob

os.chdir(os.path.dirname(__file__))

import clang.cindex

# Dont generate serialization for these enums
blacklist = ["RegexOptions", "Flags"]

enum_util_header_file = os.path.join("..", "src", "include", "duckdb", "common", "enum_util.hpp")
enum_util_source_file = os.path.join("..", "src", "common", "enum_util.cpp")


# get all the headers
hpp_files = []
for root, dirs, files in os.walk(os.path.join("..", "src")):
    for file in files:
        # Dont include the generated header itself recursively
        if file == "enum_util.hpp":
            continue

        if file.endswith(".hpp"):
            hpp_files.append(os.path.join(root, file))

# get all the enum classes
enums = []
for hpp_file in hpp_files:
    with open(hpp_file, "r") as f:
        text = f.read()
        for res in re.finditer(r"enum class (\w*)\s*:\s*(\w*)\s*{((?:\s*[^}])*)}", text, re.MULTILINE):
            file_path = os.path.relpath(hpp_file, os.path.join("..", "src")).removeprefix("include/")
            enum_name = res.group(1)

            if enum_name in blacklist:
                continue

            enum_type = res.group(2)

            enum_members = []
            # Capture All members: \w+(\s*\=\s*\w*)?
            # group one is the member name
            # group two is the member value
            # First clean group from comments
            s = res.group(3)
            s = re.sub(r"\/\/.*", "", s)
            s = re.sub(r"\/\*.*\*\/", "", s)

            enum_values = {}
            for member in re.finditer(r"(\w+)(\s*\=\s*\w*)?", s):
                member_name = member.group(1)
                if member.group(2):
                    # If the member has a value, make sure it isnt already covered by another member
                    # If it is, we cant do anything else than ignore it
                    member_value = member.group(2).strip().removeprefix("=").strip()
                    if member_value not in enum_values and member_value not in enum_members:
                        enum_members.append(member_name)    
                else:
                    enum_members.append(member_name)


            enums.append((file_path, enum_name, enum_type, enum_members))

header = """//-------------------------------------------------------------------------
// This file is automatically generated by scripts/generate_enum_util.py
// Do not edit this file manually, your changes will be overwritten
// If you want to exclude an enum from serialization, add it to the blacklist in the script
//
// Note: The generated code will only work properly if the enum is a top level item in the duckdb namespace
// If the enum is nested in a class, or in another namespace, the generated code will not compile.
// You should move the enum to the duckdb namespace, manually write a specialization or add it to the blacklist
//-------------------------------------------------------------------------\n\n
"""

# Write the enum util header
with open(enum_util_header_file, "w") as f:

    f.write(header)

    f.write('#pragma once\n\n')
    f.write('#include <stdint.h>\n\n')
    
    f.write("namespace duckdb {\n\n")
    
    f.write("""struct EnumUtil {
    // String -> Enum
    template <class T>
    static T StringToEnum(const char *value) = delete;

    // Enum -> String
    template <class T>
    static const char *EnumToString(T value) = delete;
};\n\n""")

    # Forward declare all enums
    for _, enum_name, enum_type, _ in enums:
        f.write(f"enum class {enum_name} : {enum_type};\n\n")
    f.write("\n")
    
    # Forward declare all enum serialization functions
    for _, enum_name, enum_type, _ in enums:
        f.write(f"template<>\nconst char* EnumUtil::EnumToString<{enum_name}>({enum_name} value);\n\n")
    f.write("\n")
    
    # Forward declare all enum dserialization functions
    for _, enum_name, enum_type, _ in enums:
        f.write(f"template<>\n{enum_name} EnumUtil::StringToEnum<{enum_name}>(const char *value);\n\n")
    f.write("\n")

    f.write("}\n")


with open(enum_util_source_file, "w") as f:
    f.write(header)

    f.write('#include "duckdb/common/enum_util.hpp"\n')

    # Write the includes
    for enum_path, _, _, _ in enums:
        f.write(f'#include "{enum_path}"\n')
    f.write("\n")

    f.write("namespace duckdb {\n\n")

    for _, enum_name, enum_type, enum_members in enums:
        # Write the enum from string
        f.write(f"template<>\nconst char* EnumUtil::EnumToString<{enum_name}>({enum_name} value) {{\n")
        f.write("switch(value) {\n")
        for member in enum_members:
            f.write(f"case {enum_name}::{member}: return \"{member}\";\n")
        f.write('default: throw NotImplementedException(StringUtil::Format("Enum value: \'%d\' not implemented", value));\n')
        f.write("}\n")
        f.write("}\n\n")

        # Write the string to enum
        f.write(f"template<>\n{enum_name} EnumUtil::StringToEnum<{enum_name}>(const char *value) {{\n")
        for member in enum_members:
            f.write(f'if (StringUtil::Equals(value, "{member}")) {{ return {enum_name}::{member}; }}\n')
        f.write('throw NotImplementedException(StringUtil::Format("Enum value: \'%s\' not implemented", value));\n')

        f.write("}\n\n")

    f.write("}\n\n")