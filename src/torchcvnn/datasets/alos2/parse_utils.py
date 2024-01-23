# MIT License

# Copyright (c) 2024 Jeremy Fix

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def format_key_val(key_value):
    key, value = key_value
    return key.replace("_", " ").capitalize() + f" : {value}"


def format_dictionary(record, level=0):
    strs = map(format_key_val, record.items())
    # Add heading padding
    strs = map(lambda s: " " * (level * 4) + s, strs)
    return "\n".join(strs)


def read_field(fh, start_byte, num_bytes, type_bytes, expected):
    fh.seek(start_byte)
    data_bytes = fh.read(num_bytes)
    if type_bytes[0] == "A":
        value = data_bytes.decode("ascii")
    elif type_bytes[0] == "B":
        # Binary number representation, big_endian
        value = int.from_bytes(data_bytes, "big")
    elif type_bytes[0] == "I":
        data_str = data_bytes.decode("ascii").strip()
        if len(data_str) == 0:
            value = None
        else:
            value = int(data_str)
    elif type_bytes[0] == "E":
        # This is a  Em.n format
        # Exponential notation, right fill
        m, n = type_bytes[1:].split(".")
        value = float(data_bytes.decode("ascii").strip())
    elif type_bytes[0] == "F":
        # This is a  Fm.n format
        # Exponential notation, right fill
        m, n = type_bytes[1:].split(".")
        value = float(data_bytes.decode("ascii").strip())
    if expected is not None:
        assert value == expected

    return value


def parse_from_format(
    fh, obj, descriptor_format, num_records, record_length, base_offset
):
    if num_records == 1:
        for (
            field_name,
            start_byte,
            num_bytes,
            type_bytes,
            expected,
        ) in descriptor_format:
            value = read_field(
                fh, base_offset + start_byte, num_bytes, type_bytes, expected
            )
            obj[field_name] = value
        fh.seek(base_offset + record_length)
        return base_offset + record_length
    else:
        offset = base_offset
        for _ in range(num_records):
            record = {}
            for (
                field_name,
                start_byte,
                num_bytes,
                type_bytes,
                expected,
            ) in descriptor_format:
                value = read_field(
                    fh, offset + start_byte, num_bytes, type_bytes, expected
                )
                record[field_name] = value
            obj.append(record)
            offset += record_length
        fh.seek(offset)
        return offset
