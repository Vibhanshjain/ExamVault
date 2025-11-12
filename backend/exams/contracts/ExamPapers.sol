// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ExamPapers {
    struct Paper {
        string cid;        // IPFS CID of encrypted paper
        address uploader;  // teacher wallet (not used in demo)
        uint256 time;      // block timestamp
    }

    // map subject code -> latest paper
    mapping(string => Paper) public papers;

    event PaperRecorded(string indexed s_code, string cid, address indexed uploader);

    function recordPaper(string memory s_code, string memory cid) public {
        papers[s_code] = Paper(cid, msg.sender, block.timestamp);
        emit PaperRecorded(s_code, cid, msg.sender);
    }

    function getPaper(string memory s_code) public view returns (string memory, address, uint256) {
        Paper memory p = papers[s_code];
        return (p.cid, p.uploader, p.time);
    }
}
