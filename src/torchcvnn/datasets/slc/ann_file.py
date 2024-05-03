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


class AnnFile:
    r"""

    From the documentation :

    The annotation file (.ann) is a keyword/value ASCII file in which the value on the right of the equals sign
    corresponds to the keyword on the left of the equals sign. The number of keywords may change with time, so the line
    number should not be assumed to be constant for any given keyword.

    In addition, the spacing between keywords and
    values may change. The units are given in parenthesis between the keyword and equal sign, and may change from
    annotation file to annotation file and within each annotation file.

    Comments are indicated by semicolons (;), and
    may occur at the beginning of a line, or at the middle of a line (everything after the semicolon on that line is a
    comment). The length of each text line is variable, and ends with a carriage return. There may be lines with
    just a carriage return or spaces and a carriage return.


    """

    def __init__(self, filename):
        self.filename = filename
        self.parameters = []
        self.data = self.read()

    def read(self):
        with open(self.filename, "r") as f:
            for l in f:
                l = l.strip()
                if l.startswith(";"):
                    # Skip comments
                    continue
                if len(l) != 0:
                    # Skip empty lines
                    fields = l.split(" = ")
                    if len(fields) != 2:
                        continue
                    key = fields[0].strip()
                    # key might be :
                    #     ISLR Noise Calibration Term LRTI80                       (-)
                    #     slc_1_1x1_mag.set_cols                                   (pixels)
                    # ....
                    # We split the key by spaces, take all but the last token (the units) and join them with an underscore
                    # to get a key that is easier to use
                    key = "_".join(
                        list(map(lambda s: s.strip().lower(), key.split()))[:-1]
                    ).replace(".", "_")

                    # Now we process the value, which may end with a comment
                    value = fields[1].split(";")[0].strip()
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    setattr(self, key, value)
                    self.parameters.append(key)

    def __repr__(self):
        myrep = f"AnnFile({self.filename})\n"
        for p in self.parameters:
            myrep += f"     {p} = {getattr(self, p)}\n"
        return myrep
